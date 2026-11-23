#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from tcot.cache import vertical_geometry as geom


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def indexed(data):
    result = {}
    for dataset, scenes in data.items():
        for scene, rows in scenes.items():
            for row in rows:
                if row["id"] in result:
                    raise ValueError("Duplicate cue ID")
                result[row["id"]] = (dataset, scene, row)
    return result


def selected_rows(data):
    selected = {}
    for sid, (dataset, scene, row) in indexed(data).items():
        prompts = {role: value.get("prompt", "") for role, value in (row.get("sam_prompts") or {}).items() if isinstance(value, dict)}
        if geom.vertical_question(row["question"], prompts):
            selected[sid] = (dataset, scene, row, prompts)
    return selected


def source_for_row(repo, depth_root, dataset, scene, row, prompts):
    from external_tools.o3dvqa_extraction_common import depth_path
    depth = depth_path(depth_root, dataset, scene, row["image_info"])
    if not depth.exists():
        raise FileNotFoundError(f"Missing predicted depth input: {depth}")
    masks = {}
    for role in ("subject", "reference"):
        value = (row.get("sam_prompts") or {}).get(role) or {}
        source = value.get("mask_path")
        if source:
            path = Path(source)
            if not path.is_absolute():
                path = repo / path
            if not path.exists():
                raise FileNotFoundError(f"Missing mask cache file, not a tool failure: {path}")
            masks[role] = path
    return depth, masks


def validate_record(row, original, source, config):
    expected = {key: original.get(key) for key in ("id", "question", "image_name")}
    if any(row.get(key) != value for key, value in expected.items()) or row.get("generation_config") != config:
        raise ValueError("Vertical cue input/configuration differs")
    if row.get("source") != source:
        raise ValueError("Vertical cue depth/mask provenance differs")
    ev = row["vertical3d_evidence"]
    if ev.get("uses_gt_depth") is not False or ev.get("uses_gt_pose") is not False or ev.get("version") != geom.VERSION:
        raise ValueError("Invalid vertical cue provenance")


def run(args):
    repo, root = Path(args.repo), Path(args.depth_root)
    payload = json.loads(Path(args.source_cues).read_text())
    selected = selected_rows(payload)
    if args.max_items:
        selected = dict(list(selected.items())[:args.max_items])
    config = dict(version=geom.VERSION, backend=args.backend, source_cues_path=str(Path(args.source_cues).resolve()),
                  source_cues_sha256=sha(args.source_cues), depth_root=str(root.resolve()),
                  algorithm=copy.deepcopy(geom.DEFAULTS), max_items=args.max_items,
                  numpy_version=np.__version__, code_sha256={str(Path(__file__)): sha(__file__), str(Path(geom.__file__)): sha(geom.__file__),
                  str(repo / "tcot/cache/relation_evidence_guard.py"): sha(repo / "tcot/cache/relation_evidence_guard.py")},
                  no_gt_contract="Only predicted depth+pixel K+valid_mask and cached SAM3 masks/prompts; no QA answers, pose or GT depth.")
    saved = indexed(json.loads(Path(args.out).read_text())) if Path(args.out).exists() else {}
    if not set(saved) <= set(selected):
        raise ValueError("Unexpected saved vertical IDs")
    fingerprints, sources = {}, {}
    def digest(path):
        key = str(path)
        if key not in fingerprints:
            fingerprints[key] = sha(path)
        return fingerprints[key]
    for sid, (dataset, scene, row, prompts) in selected.items():
        depth, masks = source_for_row(repo, root, dataset, scene, row, prompts)
        sources[sid] = dict(predicted_depth_path=str(depth), predicted_depth_sha256=digest(depth),
                           masks={role: dict(path=str(path), sha256=digest(path), prompt=prompts.get(role)) for role, path in masks.items()},
                           depth_semantics="camera_z_depth", intrinsics_semantics="pixel_K_at_stored_depth_resolution")
        if sid in saved:
            validate_record(saved[sid][2], row, sources[sid], config)
    print(f"[vertical-cues] backend={args.backend} selected={len(selected)}, saved={len(saved)}, dry_run={args.dry_run}", flush=True)
    if args.dry_run:
        return
    depth_cache, pair_cache = OrderedDict(), {}
    def output():
        data = {}
        for sid, (dataset, scene, _, _) in selected.items():
            if sid in saved:
                data.setdefault(dataset, {}).setdefault(scene, []).append(saved[sid][2])
        atomic(args.out, data)
    for i, (sid, (dataset, scene, row, prompts)) in enumerate(selected.items(), 1):
        if sid in saved:
            continue
        source = sources[sid]
        pair = (source["predicted_depth_path"], tuple(sorted((role, meta["sha256"], prompts.get(role)) for role, meta in source["masks"].items())))
        if pair not in pair_cache:
            if len(source["masks"]) != 2 or not all(prompts.get(role) for role in ("subject", "reference")):
                ev = dict(version=geom.VERSION, evidence_type="estimated_ground_vertical_3d", status="failed", reason="missing_pair_masks_or_prompts",
                          uses_gt_depth=False, uses_gt_pose=False)
            else:
                path = source["predicted_depth_path"]
                if path not in depth_cache:
                    with np.load(path, allow_pickle=False) as data:
                        if "depth" not in data.files or "intrinsics" not in data.files:
                            raise ValueError(f"Missing predicted depth or pixel K: {path}")
                        depth, k = data["depth"].astype(float), data["intrinsics"].astype(float)
                        valid = np.isfinite(depth) & (depth > geom.DEFAULTS["min_depth"]) & (depth < geom.DEFAULTS["max_depth"])
                        if "valid_mask" in data.files:
                            if data["valid_mask"].shape != depth.shape:
                                raise ValueError("Depth valid_mask shape mismatch")
                            valid &= data["valid_mask"].astype(bool)
                    depth_cache[path] = depth, k, valid
                    if len(depth_cache) > 4:
                        depth_cache.popitem(last=False)
                depth, k, valid = depth_cache[path]
                masks = {role: geom.load_mask(meta["path"], depth.shape) for role, meta in source["masks"].items()}
                ev = geom.build_evidence(depth, k, valid, masks, prompts)
            pair_cache[pair] = ev
        record = {key: row.get(key) for key in ("id", "question", "image_name")}
        record.update(dataset=dataset, scene=scene, image_path=row["image_info"]["image_path"],
                      object_prompts=prompts, source=source, generation_config=config, vertical3d_evidence=copy.deepcopy(pair_cache[pair]))
        saved[sid] = (dataset, scene, record)
        if i % 50 == 0:
            output()
            print(f"[progress] {i}/{len(selected)} planes/pairs={len(pair_cache)}", flush=True)
    if any(sha(path) != digest for path, digest in fingerprints.items()) or sha(args.source_cues) != config["source_cues_sha256"]:
        raise ValueError("An input changed during geometry computation; refuse to publish manifest")
    output()
    counts = Counter(row[2]["vertical3d_evidence"].get("reason") for row in saved.values())
    summary = dict(config=config, total=len(saved), usable=sum(row[2]["vertical3d_evidence"]["status"] == "ok" for row in saved.values()),
                   reasons=dict(counts), source_file_hashes=fingerprints, response_sha256=sha(args.out))
    atomic(str(args.out) + ".manifest.json", summary)
    print(json.dumps({key: summary[key] for key in ("total", "usable", "reasons")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="No-GT ground-plane vertical3D cues from existing predicted depths and SAM3 masks.")
    parser.add_argument("--repo", default=str(ROOT))
    parser.add_argument("--backend", choices=("moge2", "unidepth"), required=True)
    parser.add_argument("--source_cues")
    parser.add_argument("--depth_root")
    parser.add_argument("--out")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo)
    args.source_cues = args.source_cues or str(repo / f"target_extraction_result/relation_cues/qwen3_vl_8b_test_targets_relation_cues_sam3_{args.backend}.json")
    args.depth_root = args.depth_root or str(repo / f"estimated_depth_{args.backend}nested")
    args.out = args.out or str(repo / f"target_extraction_result/vertical3d_ground_v1/vertical3d_sam3_{args.backend}.json")
    if Path(args.out).resolve() == Path(args.source_cues).resolve():
        raise ValueError("Do not overwrite original relation cues")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
