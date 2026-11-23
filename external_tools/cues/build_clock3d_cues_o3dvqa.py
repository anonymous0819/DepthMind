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
from external_tools.cues import build_lateral3d_cues_o3dvqa as lateral
from tcot.cache import clock_geometry as geom
from tcot.cache.vertical_geometry import load_mask

ROOT = Path(__file__).resolve().parents[2]


def selected_rows(data, master):
    indexed = cache.indexed(data)
    selected = {}
    for sid, (dataset, scene, item) in master.items():
        if (item.get("qa_info") or {}).get("question_name") != "direction_data":
            continue
        d, s, row = indexed[sid]
        if (d, s, row.get("question"), row.get("image_name")) != (dataset, scene, item["query_question"], item["image_name"]):
            raise ValueError("Clock cue source/master binding differs")
        prompts = {role: obj.get("prompt", "") for role, obj in (row.get("sam_prompts") or {}).items() if isinstance(obj, dict)}
        selected[sid] = (d, s, row, prompts, geom.question_spec(row["question"], prompts))
    return selected


def validate_record(row, original, prompts, spec, source, config):
    if row.get("generation_config") != config or row.get("source") != source:
        raise ValueError("Clock cue source/config changed")
    if any(row.get(key) != original.get(key) for key in ("id", "question", "image_name")):
        raise ValueError("Clock cue QA binding changed")
    if row.get("object_prompts") != prompts or row.get("question_spec") != spec:
        raise ValueError("Clock cue object role binding changed")
    evidence = row["clock3d_evidence"]
    if evidence.get("version") != geom.VERSION or evidence.get("uses_gt_depth") is not False or evidence.get("uses_gt_pose") is not False:
        raise ValueError("Clock cues must not use GT depth/pose")
    if evidence.get("status") not in ("ok", "failed"):
        raise ValueError("Invalid clock evidence status")


def resume_configuration(saved, selected, current):
    if not set(saved) <= set(selected):
        raise ValueError("Unexpected saved clock IDs")
    if not saved:
        return copy.deepcopy(current)
    original = next(iter(saved.values()))[2]["generation_config"]
    if any(row[2].get("generation_config") != original for row in saved.values()):
        raise ValueError("Mixed clock cue configurations")
    complete = set(saved) == set(selected)
    differences = sorted(key for key in set(original) | set(current) if original.get(key) != current.get(key))
    allowed = {"code_sha256", "numpy_version"} if complete else set()
    blocking = [key for key in differences if key not in allowed]
    if blocking:
        state = "Complete cache inputs changed" if complete else "Partial cache generation configuration changed"
        raise ValueError(state + ": " + ", ".join(blocking))
    if complete and differences:
        print("[offline-validation] Complete clock cache: reusing without geometry computation; preserving original "
              f"code/NumPy provenance (NumPy {original.get('numpy_version')} -> {current.get('numpy_version')}). "
              "Differences: " + ", ".join(differences), flush=True)
    return copy.deepcopy(original)


def run(args):
    repo = Path(args.repo).resolve()
    source_path = Path(args.source_cues) if args.source_cues else repo / f"target_extraction_result/relation_cues/qwen3_vl_8b_test_targets_relation_cues_sam3_{args.backend}.json"
    master_path = Path(args.master) if args.master else Path(args.data_dir) / "test_qa_mcq.json"
    depth_root = Path(args.depth_root) if args.depth_root else repo / f"estimated_depth_{args.backend}nested"
    data = json.loads(source_path.read_text())
    master = cache.indexed(json.loads(master_path.read_text()))
    if set(cache.indexed(data)) != set(master):
        raise ValueError("Require complete source/master ID coverage")
    selected = selected_rows(data, master)
    config = dict(version=geom.VERSION, backend=args.backend, algorithm=geom.DEFAULTS, numpy_version=np.__version__,
        source_cues_path=str(source_path.resolve()), source_cues_sha256=cache.sha(source_path),
        master_path=str(master_path.resolve()), master_sha256=cache.sha(master_path), depth_root=str(depth_root),
        code_sha256={str(path.resolve()): cache.sha(path) for path in (Path(__file__), Path(geom.__file__),
            Path(cache.__file__), Path(lateral.__file__), ROOT / "tcot/cache/vertical_geometry.py")},
        contract="Predicted depth+K+valid and cached SAM3 only; no GT pose/depth. A reference/B target, pitch-only relative vector; no bearing or clock answer.")
    saved = cache.indexed(json.loads(Path(args.out).read_text())) if Path(args.out).exists() else {}
    config = resume_configuration(saved, selected, config)
    manifest_path = Path(args.out + ".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("response_sha256") != cache.sha(args.out) or manifest.get("total") != len(saved)
                or manifest.get("config") != config):
            raise ValueError("Existing clock cache/manifest fingerprint or provenance differs")
    hashes, sources = {}, {}
    def digest(path):
        key = str(Path(path).resolve())
        if key not in hashes:
            hashes[key] = cache.sha(path)
        return hashes[key]
    for sid, (d, s, row, prompts, spec) in selected.items():
        sources[sid] = lateral.source_for_row(repo, depth_root, d, s, row, spec, digest)
        if sid in saved:
            validate_record(saved[sid][2], row, prompts, spec, sources[sid], config)
    print(f"[clock-cues] selected={len(selected)} saved={len(saved)} backend={args.backend}", flush=True)
    if args.dry_run:
        return
    depths = OrderedDict()
    def output():
        payload = {}
        for sid, (d, s, _, _, _) in selected.items():
            if sid in saved:
                payload.setdefault(d, {}).setdefault(s, []).append(saved[sid][2])
        cache.atomic(args.out, payload)
    changed = False
    for number, (sid, (d, s, row, prompts, spec)) in enumerate(selected.items(), 1):
        if sid in saved:
            continue
        source = sources[sid]
        path = source["predicted_depth_path"]
        if path not in depths:
            with np.load(path, allow_pickle=False) as npz:
                if "depth" not in npz.files or "intrinsics" not in npz.files:
                    raise ValueError("Predicted depth/K missing")
                depth, k = npz["depth"].astype(float), npz["intrinsics"].astype(float)
                if depth.ndim != 2:
                    raise ValueError("Depth must be 2D")
                valid = np.isfinite(depth) & (depth > geom.DEFAULTS["min_depth"]) & (depth < geom.DEFAULTS["max_depth"])
                if "valid_mask" in npz.files:
                    if npz["valid_mask"].shape != depth.shape:
                        raise ValueError("Depth valid_mask shape differs")
                    valid &= npz["valid_mask"].astype(bool)
                from tcot.cache.vertical_geometry import project
                project(depth, k, np.zeros_like(valid))
            depths[path] = depth, k, valid
            if len(depths) > 4:
                depths.popitem(last=False)
        depth, k, valid = depths[path]
        masks = {role: load_mask(obj["path"], depth.shape) for role, obj in source["masks"].items()}
        evidence = geom.build_evidence(depth, k, valid, masks, prompts, spec)
        rec = {key: row.get(key) for key in ("id", "question", "image_name")}
        rec.update(dataset=d, scene=s, object_prompts=prompts, question_spec=spec, source=source,
                   generation_config=copy.deepcopy(config), clock3d_evidence=evidence)
        saved[sid] = d, s, rec
        changed = True
        if number % 20 == 0:
            output()
    if any(cache.sha(path) != value for path, value in hashes.items()) or cache.sha(source_path) != config["source_cues_sha256"] or cache.sha(master_path) != config["master_sha256"]:
        raise ValueError("Inputs changed during cue computation")
    if changed or not Path(args.out).exists():
        output()
    manifest = dict(config=config, total=len(saved), usable=sum(row[2]["clock3d_evidence"]["status"] == "ok" for row in saved.values()),
                    reasons=dict(Counter(row[2]["clock3d_evidence"]["reason"] for row in saved.values())),
                    source_file_hashes=hashes, response_sha256=cache.sha(args.out))
    path = Path(args.out + ".manifest.json")
    if not path.exists() or json.loads(path.read_text()) != manifest:
        cache.atomic(str(path), manifest)
    print(json.dumps({key: manifest[key] for key in ("total", "usable", "reasons")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Predicted pitch-only horizontal relative3D clock cues; no GT")
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
    args.out = args.out or str(Path(args.repo) / f"target_extraction_result/clock_horizontal3d_v1/clock3d_sam3_{args.backend}.json")
    path = Path(args.out).resolve()
    if path.parent != Path(args.repo) / "target_extraction_result/clock_horizontal3d_v1":
        parser.error("Use the isolated clock_horizontal3d_v1 output directory")
    if args.dry_run:
        run(args)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)


if __name__ == "__main__":
    main()
