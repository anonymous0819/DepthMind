#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from external_tools.cues import build_vertical3d_cues_o3dvqa as cache
from tcot.cache import lateral_geometry as geom
from tcot.cache.vertical_geometry import load_mask

ROOT = Path(__file__).resolve().parents[2]


def selected_rows(data, metric_ids):
    result = {}
    for sid, (dataset, scene, row) in cache.indexed(data).items():
        prompts = {role: obj.get("prompt", "") for role, obj in (row.get("sam_prompts") or {}).items() if isinstance(obj, dict)}
        spec = geom.question_spec(row["question"], prompts)
        if sid not in metric_ids and spec is not None:
            result[sid] = (dataset, scene, row, prompts, spec)
    return result


def source_for_row(repo, depth_root, dataset, scene, row, spec, digest):
    depth, masks = cache.source_for_row(repo, depth_root, dataset, scene, row, {})
    masks = {role: path for role, path in masks.items() if role in spec["required_roles"]}
    return dict(predicted_depth_path=str(depth), predicted_depth_sha256=digest(depth),
                masks={role: dict(path=str(path), sha256=digest(path), prompt=row["sam_prompts"][role]["prompt"]) for role, path in masks.items()},
                depth_semantics="camera_z_depth", intrinsics_semantics="pixel_K_at_stored_depth_resolution")


def validate_record(record, original, prompts, spec, source, config):
    if record.get("generation_config") != config or record.get("source") != source:
        raise ValueError("Lateral cues source/configuration changed")
    if any(record.get(key) != original.get(key) for key in ("id", "question", "image_name")):
        raise ValueError("Lateral cues QA binding changed")
    if record.get("object_prompts") != prompts or record.get("question_spec") != spec:
        raise ValueError("Lateral cues object/relationship binding changed")
    ev = record["lateral3d_evidence"]
    if ev.get("version") != geom.VERSION or ev.get("uses_gt_depth") is not False or ev.get("uses_gt_pose") is not False:
        raise ValueError("Lateral evidence must not use GT depth/pose")


def run(args):
    repo, depth_root = Path(args.repo), Path(args.depth_root)
    data = json.loads(Path(args.source_cues).read_text())
    master = cache.indexed(json.loads(Path(args.master).read_text()))
    metric = {sid for sid, (_, _, row) in master.items() if row.get("mcq_tier") in ("T3a", "T3b")}
    if set(cache.indexed(data)) != set(master):
        raise ValueError("Require complete original cue/master coverage")
    selected = selected_rows(data, metric)
    config = dict(version=geom.VERSION, backend=args.backend, algorithm=geom.DEFAULTS, numpy_version=np.__version__,
        source_cues_path=str(Path(args.source_cues).resolve()), source_cues_sha256=cache.sha(args.source_cues),
        master_path=str(Path(args.master).resolve()), master_sha256=cache.sha(args.master), depth_root=str(depth_root.resolve()),
        code_sha256={str(path): cache.sha(path) for path in (Path(__file__), Path(geom.__file__), ROOT / "tcot/cache/vertical_geometry.py")},
        contract="Only predicted depth+K+valid_mask and cached SAM3 masks; question-only relation selection, no question_name/GT selection.")
    saved = cache.indexed(json.loads(Path(args.out).read_text())) if Path(args.out).exists() else {}
    if not set(saved) <= set(selected):
        raise ValueError("Unexpected saved lateral IDs")
    if saved:
        original = next(iter(saved.values()))[2]["generation_config"]
        if any(row[2]["generation_config"] != original for row in saved.values()):
            raise ValueError("Mixed historical cue generation configurations")
        old_noncode = {key: value for key, value in original.items() if key != "code_sha256"}
        new_noncode = {key: value for key, value in config.items() if key != "code_sha256"}
        if old_noncode != new_noncode or set(saved) != set(selected) and original != config:
            raise ValueError("Cue inputs changed or partial generation code differs")
        if set(saved) == set(selected):
            config = copy.deepcopy(original)
    hashes, sources = {}, {}
    def digest(path):
        key = str(path)
        if key not in hashes:
            hashes[key] = cache.sha(path)
        return hashes[key]
    for sid, (dataset, scene, row, prompts, spec) in selected.items():
        sources[sid] = source_for_row(repo, depth_root, dataset, scene, row, spec, digest)
        if sid in saved:
            validate_record(saved[sid][2], row, prompts, spec, sources[sid], config)
    print(f"[lateral-cues] selected={len(selected)} saved={len(saved)} backend={args.backend}", flush=True)
    if args.dry_run:
        return
    depth_cache, geometry_cache = OrderedDict(), {}
    def output():
        result = {}
        for sid, (dataset, scene, _, _, _) in selected.items():
            if sid in saved:
                result.setdefault(dataset, {}).setdefault(scene, []).append(saved[sid][2])
        cache.atomic(args.out, result)
    changed = False
    for i, (sid, (dataset, scene, row, prompts, spec)) in enumerate(selected.items(), 1):
        if sid in saved:
            continue
        source = sources[sid]
        key = (source["predicted_depth_path"], tuple(sorted((role, obj["sha256"], obj["prompt"]) for role, obj in source["masks"].items())), spec["comparison"], spec["applicable"])
        if key not in geometry_cache:
            path = source["predicted_depth_path"]
            if path not in depth_cache:
                with np.load(path, allow_pickle=False) as npz:
                    if "depth" not in npz.files or "intrinsics" not in npz.files:
                        raise ValueError("Missing depth or intrinsics is not a tool failure")
                    depth, k = npz["depth"].astype(float), npz["intrinsics"].astype(float)
                    if depth.ndim != 2:
                        raise ValueError("Depth must be 2D")
                    valid = np.isfinite(depth) & (depth > geom.DEFAULTS["min_depth"]) & (depth < geom.DEFAULTS["max_depth"])
                    if "valid_mask" in npz.files:
                        if npz["valid_mask"].shape != depth.shape:
                            raise ValueError("Depth valid_mask shape differs")
                        valid &= npz["valid_mask"].astype(bool)
                depth_cache[path] = depth, k, valid
                if len(depth_cache) > 4:
                    depth_cache.popitem(last=False)
            depth, k, valid = depth_cache[path]
            masks = {role: load_mask(obj["path"], depth.shape) for role, obj in source["masks"].items()}
            geometry_cache[key] = geom.build_evidence(depth, k, valid, masks, prompts, spec)
        record = {name: row.get(name) for name in ("id", "question", "image_name")}
        record.update(dataset=dataset, scene=scene, object_prompts=prompts, question_spec=spec, source=source,
                      generation_config=copy.deepcopy(config), lateral3d_evidence=copy.deepcopy(geometry_cache[key]))
        saved[sid] = (dataset, scene, record)
        changed = True
        if i % 100 == 0:
            output()
    if any(cache.sha(path) != value for path, value in hashes.items()) or cache.sha(args.source_cues) != config["source_cues_sha256"] or cache.sha(args.master) != config["master_sha256"]:
        raise ValueError("Inputs changed during cue computation")
    if changed or not Path(args.out).exists():
        output()
    summary = dict(config=config, total=len(saved), usable=sum(r[2]["lateral3d_evidence"]["status"] == "ok" for r in saved.values()),
                   reasons=dict(Counter(r[2]["lateral3d_evidence"]["reason"] for r in saved.values())),
                   source_file_hashes=hashes, response_sha256=cache.sha(args.out))
    target = Path(args.out + ".manifest.json")
    if not target.exists() or json.loads(target.read_text()) != summary:
        cache.atomic(target, summary)
    print(json.dumps({key: summary[key] for key in ("total", "usable", "reasons")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Compute camera-lateral visible 3D centroids without GT or ground-plane estimation.")
    parser.add_argument("--repo", default=str(ROOT))
    parser.add_argument("--backend", choices=("moge2", "unidepth"), required=True)
    parser.add_argument("--data_dir", default="/root/src/UAV_reasoning/O3DVQA_v2")
    parser.add_argument("--source_cues")
    parser.add_argument("--master")
    parser.add_argument("--depth_root")
    parser.add_argument("--out")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    args.repo = str(Path(args.repo).resolve())
    repo = Path(args.repo)
    args.source_cues = args.source_cues or str(repo / f"target_extraction_result/relation_cues/qwen3_vl_8b_test_targets_relation_cues_sam3_{args.backend}.json")
    args.master = args.master or str(Path(args.data_dir) / "test_qa_mcq.json")
    args.depth_root = args.depth_root or str(repo / f"estimated_depth_{args.backend}nested")
    args.out = args.out or str(repo / f"target_extraction_result/lateral3d_center_v1/lateral3d_sam3_{args.backend}.json")
    if Path(args.out).resolve() in {Path(args.source_cues).resolve(), Path(args.master).resolve()}:
        raise ValueError("Do not overwrite historical inputs")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
