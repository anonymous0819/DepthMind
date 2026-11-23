#!/usr/bin/env python3
"""Build no-GT Open3D-VQA relation cues from SAM3.1 masks + estimated depth.

This is the no-GT replacement for the old SAM-depth cue extraction path:
    - grounding/masks come from cached SAM3.1 predictions only;
    - depth comes from MoGe2 / UniDepth estimated-depth npz caches;
    - GT bbox, GT mask, GT intrinsics and GT depth are never loaded.

GroundingDINO is intentionally not used here, to match the existing tool-first
geometry tool protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

O3D_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(O3D_ROOT))
from external_tools.o3dvqa_extraction_common import depth_path, load_mask_from_npz, load_sam3_index, normalize_relpath, resize_depth_to, resize_mask_to
from external_tools.segmentation.sam_cue_helpers import INVALID_DEPTH_THRESHOLD, collect_prompts, compute_mask_item, count_items, iter_nested_items
EST_DEPTH_ROOTS = {name: (O3D_ROOT / f"estimated_depth_{name}nested", (378, 504) if name == "da3" else (480, 640)) for name in ("da3", "moge2", "unidepth")}

DEFAULT_DATA_DIR = "/root/src/UAV_reasoning/O3DVQA_v2"
DEFAULT_TARGET_FILE = str(
    O3D_ROOT / "target_extraction_result/qwen3_vl_8b_test_targets.json"
)
DEFAULT_RESULT_DIR = str(O3D_ROOT / "target_extraction_result/relation_cues")
MIN_VALID_DEPTH = 0.5
CENTER_DECIMALS = 4
DEPTH_DECIMALS = 3


def _nested_results() -> defaultdict:
    return defaultdict(lambda: defaultdict(list))


def default_result_path(target_file: str, depth_backend: str, mask_source: str) -> str:
    stem = Path(target_file).stem
    return str(
        Path(DEFAULT_RESULT_DIR)
        / f"{stem}_relation_cues_{mask_source}_{depth_backend}.json"
    )


def load_existing_results(result_path: str) -> Tuple[defaultdict, set[str]]:
    if not os.path.exists(result_path):
        return _nested_results(), set()
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = _nested_results()
    finished_ids: set[str] = set()
    for dataset, scenes in data.items():
        if not isinstance(scenes, dict):
            continue
        for scene, items in scenes.items():
            if not isinstance(items, list):
                continue
            results[dataset][scene].extend(items)
            for item in items:
                item_id = item.get("id") if isinstance(item, dict) else None
                if item_id:
                    finished_ids.add(str(item_id))
    return results, finished_ids


def save_results(results: defaultdict, result_path: str) -> None:
    out_path = Path(result_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {dataset: dict(scenes) for dataset, scenes in results.items()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def build_est_depth_path(
    depth_root: Path,
    dataset: str,
    scene: str,
    image_info: Dict[str, Any],
) -> Path:
    image_path = image_info.get("image_path")
    if not image_path:
        raise ValueError("missing image_info.image_path")
    image_stem = Path(normalize_relpath(str(image_path))).stem
    return depth_root / dataset / scene / "rgb" / f"{image_stem}.npz"


def load_estimated_depth(
    depth_backend: str,
    depth_root: Path,
    dataset: str,
    scene: str,
    image_info: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Path]:
    depth_path = build_est_depth_path(depth_root, dataset, scene, image_info)
    if not depth_path.exists():
        raise FileNotFoundError(f"estimated depth file not found: {depth_path}")
    data = np.load(depth_path)
    if "depth" not in data.files:
        raise KeyError(f"missing 'depth' key in {depth_path}")
    depth = data["depth"].astype(np.float32)
    intrinsics = data["intrinsics"].astype(np.float64) if "intrinsics" in data.files else None
    valid_mask = data["valid_mask"].astype(bool) if "valid_mask" in data.files else None

    expected_hw = EST_DEPTH_ROOTS.get(depth_backend, (None, depth.shape))[1]
    if expected_hw and tuple(depth.shape) != tuple(expected_hw):
        depth = resize_depth_to(depth, expected_hw)
        if valid_mask is not None:
            valid_mask = resize_mask_to(valid_mask, expected_hw)
    return depth, intrinsics, valid_mask, depth_path


def compute_global_depth_stats(
    depth: np.ndarray,
    estimated_valid_mask: Optional[np.ndarray],
    invalid_depth_threshold: float,
) -> Tuple[np.ndarray, Dict[str, Optional[float]]]:
    valid = (
        np.isfinite(depth)
        & (depth > MIN_VALID_DEPTH)
        & (depth < float(invalid_depth_threshold))
    )
    if estimated_valid_mask is not None:
        if estimated_valid_mask.shape != depth.shape:
            estimated_valid_mask = resize_mask_to(estimated_valid_mask, depth.shape)
        valid &= estimated_valid_mask.astype(bool)

    values = depth[valid]
    if values.size == 0:
        return valid, {
            "valid_depth_ratio": 0.0,
            "median_depth": None,
            "p10_depth": None,
            "p33_depth": None,
            "p66_depth": None,
            "p90_depth": None,
        }

    p10, p33, p50, p66, p90 = np.percentile(values, [10, 33, 50, 66, 90])
    return valid, {
        "valid_depth_ratio": float(valid.mean()),
        "median_depth": float(p50),
        "p10_depth": float(p10),
        "p33_depth": float(p33),
        "p66_depth": float(p66),
        "p90_depth": float(p90),
    }


def _round_float(value: Any, decimals: int) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


def _round_float_list(values: Any, decimals: int) -> Optional[list[float]]:
    if not isinstance(values, (list, tuple)):
        return None
    out: list[float] = []
    for value in values:
        rounded = _round_float(value, decimals)
        if rounded is None:
            return None
        out.append(rounded)
    return out


def normalize_bbox_xyxy(
    bbox_xyxy: Any,
    image_width: int,
    image_height: int,
) -> Optional[list[float]]:
    if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    denom_x = max(1, image_width - 1)
    denom_y = max(1, image_height - 1)
    return [
        max(0.0, min(1.0, x1 / denom_x)),
        max(0.0, min(1.0, y1 / denom_y)),
        max(0.0, min(1.0, x2 / denom_x)),
        max(0.0, min(1.0, y2 / denom_y)),
    ]


def compact_object_geometry(target: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if target is None:
        return None
    obj: Dict[str, Any] = {
        "role": target.get("role"),
        "prompt": target.get("prompt"),
        "center_xy_normalized": _round_float_list(
            target.get("center_xy_normalized"), CENTER_DECIMALS,
        ),
        "bbox_xyxy_normalized": _round_float_list(
            target.get("bbox_xyxy_normalized"), CENTER_DECIMALS,
        ),
        "depth_mean": _round_float(target.get("mean_depth"), DEPTH_DECIMALS),
        "valid_depth_ratio": _round_float(
            target.get("valid_depth_ratio"), CENTER_DECIMALS,
        ),
    }
    return {k: v for k, v in obj.items() if v is not None}


def build_object_geometry_prompt_cues(
    objects: Dict[str, Optional[Dict[str, Any]]],
) -> list[str]:
    cues: list[str] = []
    for role in ("subject", "reference"):
        obj = objects.get(role)
        if not obj:
            continue
        prompt = obj.get("prompt") or role
        center = obj.get("center_xy_normalized")
        bbox = obj.get("bbox_xyxy_normalized")
        depth_mean = obj.get("depth_mean")
        valid_depth_ratio = obj.get("valid_depth_ratio")
        pieces = [f"{role}='{prompt}'"]
        if center is not None:
            pieces.append(f"center_xy_normalized={center}")
        if bbox is not None:
            pieces.append(f"bbox_xyxy_normalized={bbox}")
        if depth_mean is not None:
            pieces.append(f"depth_mean={depth_mean}")
        if valid_depth_ratio is not None:
            pieces.append(f"valid_depth_ratio={valid_depth_ratio}")
        cues.append("- " + ", ".join(pieces) + ".")
    return cues


def build_mask_index(mask_source: str, mask_file: str) -> Dict[str, Dict[str, str]]:
    if mask_source == "sam3":
        return load_sam3_index(mask_file)
    raise ValueError(f"unknown mask_source: {mask_source}; only 'sam3' is supported")


def compute_role_target(
    role: str,
    prompt: str,
    mask_path: Optional[str],
    depth: np.ndarray,
    valid_depth_mask: np.ndarray,
    global_depth_stats: Dict[str, Optional[float]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    prompt_result: Dict[str, Any] = {
        "prompt": prompt,
        "mask_path": mask_path,
        "num_masks": 0,
        "items": [],
        "best_mask_idx": None,
        "status": "missing_mask_path" if not mask_path else "ok",
    }
    if not mask_path:
        return None, prompt_result, "missing_mask_path"

    mask = load_mask_from_npz(mask_path)
    if mask is None:
        prompt_result["status"] = "bad_mask_npz"
        return None, prompt_result, "bad_mask_npz"
    mask = resize_mask_to(mask, depth.shape)

    item = compute_mask_item(
        mask=mask,
        depth=depth,
        valid_depth_mask=valid_depth_mask,
        global_stats=global_depth_stats,
        box_xywh_normalized=None,
        score=None,
        idx=0,
    )
    image_height, image_width = mask.shape
    bbox_xyxy_normalized = normalize_bbox_xyxy(
        item.get("bbox_xyxy"), image_width=image_width, image_height=image_height,
    )
    if bbox_xyxy_normalized is not None:
        item["bbox_xyxy_normalized"] = bbox_xyxy_normalized
    prompt_result.update({
        "num_masks": 1,
        "items": [item],
        "best_mask_idx": 0,
        "status": "ok",
    })
    target = {
        "role": role,
        "prompt": prompt,
        "mask_path": mask_path,
        "best_mask_idx": 0,
        **item,
    }
    return target, prompt_result, None


def process_item(
    item: Dict[str, Any],
    dataset: str,
    scene: str,
    args: argparse.Namespace,
    mask_index: Dict[str, Dict[str, str]],
    depth_cache: Dict[Tuple[str, str, str], Tuple[np.ndarray, Optional[np.ndarray], Path]],
) -> Dict[str, Any]:
    item_id = str(item.get("id"))
    image_info = item.get("image_info", {}) or {}
    extraction = item.get("target_extraction", {}) or {}
    prompts = collect_prompts(extraction)
    image_stem = Path(normalize_relpath(str(image_info.get("image_path", "")))).stem
    cache_key = (dataset, scene, image_stem)

    if cache_key in depth_cache:
        depth, estimated_valid_mask, depth_path = depth_cache[cache_key]
    else:
        depth, _intrinsics, estimated_valid_mask, depth_path = load_estimated_depth(
            depth_backend=args.depth_backend,
            depth_root=Path(args.depth_root),
            dataset=dataset,
            scene=scene,
            image_info=image_info,
        )
        depth_cache[cache_key] = (depth, estimated_valid_mask, depth_path)

    valid_depth_mask, global_depth_stats = compute_global_depth_stats(
        depth=depth,
        estimated_valid_mask=estimated_valid_mask,
        invalid_depth_threshold=args.invalid_depth_threshold,
    )

    masks_for_item = mask_index.get(item_id, {})
    prompt_results: Dict[str, Dict[str, Any]] = {}
    grounded_targets: Dict[str, Optional[Dict[str, Any]]] = {
        "subject": None,
        "reference": None,
    }
    fail_reasons: list[str] = []

    for role in ("subject", "reference"):
        prompt = prompts.get(role)
        if not prompt:
            continue
        target, prompt_result, reason = compute_role_target(
            role=role,
            prompt=prompt,
            mask_path=masks_for_item.get(role),
            depth=depth,
            valid_depth_mask=valid_depth_mask,
            global_depth_stats=global_depth_stats,
        )
        prompt_results[role] = prompt_result
        grounded_targets[role] = target
        if reason:
            fail_reasons.append(f"{role}:{reason}")

    relative_relations = []
    subject = grounded_targets.get("subject")
    reference = grounded_targets.get("reference")
    object_geometry = {
        "subject": compact_object_geometry(subject),
        "reference": compact_object_geometry(reference),
    }
    prompt_cues = build_object_geometry_prompt_cues(object_geometry)

    has_subject_prompt = bool(prompts.get("subject"))
    subject_ok = grounded_targets.get("subject") is not None
    reference_needed = bool(prompts.get("reference"))
    reference_ok = (not reference_needed) or grounded_targets.get("reference") is not None
    if not has_subject_prompt:
        status = "no_subject_prompt"
    elif subject_ok and reference_ok:
        status = "ok"
    elif subject_ok:
        status = "partial"
    else:
        status = "failed"

    source = {
        "uses_gt": False,
        "mask_source": args.mask_source,
        "depth_backend": args.depth_backend,
        "depth_path": str(depth_path),
        "depth_root": str(args.depth_root),
        "mask_index_size": len(mask_index),
    }

    return {
        "id": item_id,
        "dataset": dataset,
        "scene": scene,
        "image_name": item.get("image_name"),
        "image_info": image_info,
        "qa_info": item.get("qa_info", {}),
        "question": item.get("question") or item.get("query_question"),
        "target_extraction": extraction,
        "status": status,
        "fail_reasons": fail_reasons,
        "image_size": [int(depth.shape[1]), int(depth.shape[0])],
        "depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "global_depth_stats": global_depth_stats,
        "sam_prompts": prompt_results,
        "grounded_targets": grounded_targets,
        "object_geometry": object_geometry,
        "relative_relations": relative_relations,
        "prompt_cues": prompt_cues,
        "relation_evidence": {
            "name": "relation_evidence",
            "status": status,
            "evidence_type": "object_geometry",
            "objects": object_geometry,
            "prompt_cues": prompt_cues,
            "source": source,
        },
        "source": source,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract Open3D-VQA relation cues from cached SAM3.1 masks "
            "and MoGe2/UniDepth estimated depth. No GT depth/K/mask/bbox "
            "is used; GroundingDINO is not used."
        )
    )
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--target_file", type=str, default=DEFAULT_TARGET_FILE)
    p.add_argument("--result_path", type=str, default=None)
    p.add_argument("--depth_backend", choices=["da3", "moge2", "unidepth"], required=True)
    p.add_argument("--depth_root", type=str, default=None)
    p.add_argument("--mask_source", choices=["sam3"], default="sam3")
    p.add_argument("--mask_file", required=True, help="Predicted SAM mask index JSON")
    p.add_argument("--invalid_depth_threshold", type=float, default=INVALID_DEPTH_THRESHOLD)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if args.depth_root is None:
        args.depth_root = str(EST_DEPTH_ROOTS[args.depth_backend][0])
    if args.result_path is None:
        args.result_path = default_result_path(
            args.target_file, args.depth_backend, args.mask_source)
    return args


def main() -> None:
    args = parse_args()
    with open(args.target_file, "r", encoding="utf-8") as f:
        target_data = json.load(f)

    if args.resume:
        results, finished_ids = load_existing_results(args.result_path)
    else:
        results, finished_ids = _nested_results(), set()

    mask_index = build_mask_index(args.mask_source, args.mask_file)
    print("[relation-cues] no-GT extraction", flush=True)
    print(f"  target_file   = {args.target_file}", flush=True)
    print(f"  result_path   = {args.result_path}", flush=True)
    print(f"  mask_source   = {args.mask_source} ({len(mask_index)} indexed items)", flush=True)
    print(f"  depth_backend = {args.depth_backend}", flush=True)
    print(f"  depth_root    = {args.depth_root}", flush=True)

    total = count_items(target_data)
    if args.max_samples is not None and args.max_samples > 0:
        total = min(total, args.max_samples)

    depth_cache: Dict[Tuple[str, str, str], Tuple[np.ndarray, Optional[np.ndarray], Path]] = {}
    status_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    processed_count = 0

    pbar = tqdm(total=total, desc=f"relation-cues[{args.depth_backend}/{args.mask_source}]")
    for dataset, scene, _index, item in iter_nested_items(target_data):
        item_id = str(item.get("id"))
        if item_id in finished_ids:
            pbar.update(1)
            continue

        try:
            record = process_item(
                item=item,
                dataset=dataset,
                scene=scene,
                args=args,
                mask_index=mask_index,
                depth_cache=depth_cache,
            )
        except Exception as exc:  # noqa: BLE001 - preserve one record per item
            record = {
                "id": item_id,
                "dataset": dataset,
                "scene": scene,
                "image_name": item.get("image_name"),
                "image_info": item.get("image_info", {}),
                "qa_info": item.get("qa_info", {}),
                "question": item.get("question") or item.get("query_question"),
                "target_extraction": item.get("target_extraction", {}),
                "status": "error",
                "error": repr(exc),
                "prompt_cues": [],
                "relation_evidence": {
                    "name": "relation_evidence",
                    "status": "error",
                    "prompt_cues": [],
                    "source": {
                        "uses_gt": False,
                        "mask_source": args.mask_source,
                        "depth_backend": args.depth_backend,
                        "depth_root": str(args.depth_root),
                    },
                },
            }

        results[dataset][scene].append(record)
        finished_ids.add(item_id)
        status_counter[str(record.get("status"))] += 1
        for reason in record.get("fail_reasons", []) or []:
            reason_counter[str(reason)] += 1

        processed_count += 1
        pbar.update(1)
        if processed_count % max(1, args.save_every) == 0:
            save_results(results, args.result_path)
        if args.max_samples is not None and args.max_samples > 0 and processed_count >= args.max_samples:
            break

    pbar.close()
    save_results(results, args.result_path)
    print(f"[done] wrote {processed_count} new records to {args.result_path}", flush=True)
    print(f"[done] depth cache entries: {len(depth_cache)}", flush=True)
    print(f"[done] status counts: {dict(status_counter)}", flush=True)
    if reason_counter:
        print(f"[done] failure reasons: {dict(reason_counter.most_common(20))}", flush=True)


if __name__ == "__main__":
    main()
