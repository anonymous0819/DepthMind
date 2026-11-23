#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dronepur_common import file_digest, write_json_atomic
from external_tools.depth.extract_dronepur_da3_depth import save_depth
from external_tools.o3dvqa_extraction_common import build_image_path, iter_items
from external_tools.segmentation.sam_cue_helpers import (
    build_sam_query_candidates, collect_prompts, load_sam3_predictor, run_sam_prompt,
)


def predict_masks(predictor, image_path, prompts, image_hw, threshold, enable_query_fallback=True):
    results = {}
    if not prompts:
        return results
    session = predictor.handle_request({"type": "start_session", "resource_path": str(image_path)})
    session_id = session["session_id"]
    try:
        for role, prompt in prompts.items():
            attempts = []
            for query in build_sam_query_candidates(prompt):
                masks, boxes, scores = run_sam_prompt(predictor, session_id, query, 0, threshold, image_hw)
                attempts.append(dict(query_prompt=query, num_masks=int(masks.shape[0])))
                if masks.shape[0] > 0 or not enable_query_fallback:
                    break
            results[role] = (masks, boxes, scores, query, attempts)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract SAM3 masks using question-only target phrases; no GT depth, boxes or masks")
    parser.add_argument("--target_file", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument("--result_path", type=Path, required=True)
    parser.add_argument("--sam3_src", required=True)
    parser.add_argument("--sam3_checkpoint", type=Path, required=True)
    parser.add_argument("--sam3_version", default="sam3.1")
    parser.add_argument("--sam3_model_max_num_objects", type=int, default=16)
    parser.add_argument("--sam3_multiplex_count", type=int, default=16)
    parser.add_argument("--output_prob_thresh", type=float, default=.5)
    parser.add_argument("--max_masks_per_prompt", type=int, default=3)
    parser.add_argument("--disable_query_fallback", dest="enable_query_fallback", action="store_false", default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.output_prob_thresh <= 1:
        parser.error("output_prob_thresh must be in [0,1]")
    if min(args.sam3_model_max_num_objects, args.sam3_multiplex_count, args.max_masks_per_prompt) < 1:
        parser.error("SAM object limits must be positive")
    for output in (args.result_path, args.mask_dir):
        if output.resolve().is_relative_to(args.data_dir.resolve()):
            parser.error("Outputs must be outside the source dataset")
    if args.result_path.resolve() == args.target_file.resolve():
        parser.error("Do not overwrite input target extraction")
    stat = args.sam3_checkpoint.stat()
    config = dict(target_sha256=file_digest(args.target_file), code_sha256=file_digest(__file__),
                  helpers_sha256=file_digest(run_sam_prompt.__code__.co_filename),
                  sam3_src=str(Path(args.sam3_src).resolve()), checkpoint=str(args.sam3_checkpoint.resolve()),
                  checkpoint_size=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns,
                  version=args.sam3_version, max_objects=args.sam3_model_max_num_objects,
                  multiplex_count=args.sam3_multiplex_count, threshold=args.output_prob_thresh,
                  max_masks_per_prompt=args.max_masks_per_prompt,
                  mask_dir=str(args.mask_dir.resolve()), uses_gt=False, prompt_fallback=args.enable_query_fallback,
                  session_policy="one_per_sample_continuous_roles")
    sidecar = args.result_path.with_name(args.result_path.name + ".manifest.json")
    saved = {}
    if args.result_path.exists() or sidecar.exists():
        if not args.resume:
            raise FileExistsError("Use --resume or new result_path and mask_dir")
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if previous["config"] != config:
            raise ValueError("Mask extraction identity changed; use a new output")
        data = json.loads(args.result_path.read_text(encoding="utf-8"))
        for dataset, scene, row in iter_items(data):
            sid = str(row["id"])
            if sid in saved:
                raise ValueError(f"Duplicate saved ID: {sid}")
            saved[sid] = (dataset, scene, row)
    payload = json.loads(args.target_file.read_text(encoding="utf-8"))
    rows, ids = list(iter_items(payload)), set()
    for _, _, row in rows:
        sid = str(row["id"])
        if sid in ids:
            raise ValueError(f"Duplicate target ID: {sid}")
        ids.add(sid)
    if set(saved) - ids:
        raise ValueError("Saved masks include IDs outside target input")

    def save():
        data = {}
        for dataset, scene, source in rows:
            item = saved.get(str(source["id"]))
            if item is not None:
                data.setdefault(dataset, {}).setdefault(scene, []).append(item[2])
        write_json_atomic(args.result_path, data)
        write_json_atomic(sidecar, dict(config=config, expected_samples=len(rows),
                                       completed_samples=len(saved), complete=len(saved) == len(rows)))

    predictor = None
    try:
        save()
        from PIL import Image

        for dataset, scene, row in rows:
            sid = str(row["id"])
            image_path = Path(build_image_path(args.data_dir, dataset, scene, row["image_info"]))
            rgb_sha = file_digest(image_path)
            if sid in saved:
                previous = saved[sid][2]
                if previous["rgb_sha256"] != rgb_sha:
                    raise ValueError(f"Source image changed: {sid}")
                for obj in previous["sam_prompts"].values():
                    if obj["mask_path"] and file_digest(obj["mask_path"]) != obj["mask_sha256"]:
                        raise ValueError(f"Cached masks changed: {sid}")
                continue
            prompts = collect_prompts(row.get("target_extraction") or {})
            if predictor is None and prompts:
                predictor = load_sam3_predictor(args)
            with Image.open(image_path) as image:
                image_hw = (image.height, image.width)
            predictions = predict_masks(predictor, image_path, prompts, image_hw, args.output_prob_thresh, args.enable_query_fallback)
            objects = {}
            for role, (masks, boxes, scores, query, attempts) in predictions.items():
                limit = args.max_masks_per_prompt
                masks, boxes, scores = masks[:limit], boxes[:limit], scores[:limit]
                path = None
                if len(masks):
                    import hashlib
                    key = hashlib.sha256((sid + "\0" + role).encode()).hexdigest()[:24]
                    path = args.mask_dir.resolve() / f"{key}.npz"
                    save_depth(path, dict(masks=masks.astype("uint8")), rgb_sha, str(image_path), config)
                objects[role] = dict(prompt=prompts[role], sam_query_prompt=query, query_attempts=attempts,
                                     num_masks=len(masks), mask_path=str(path) if path else None,
                                     mask_sha256=file_digest(path) if path else None,
                                     boxes_xywh_normalized=boxes, scores=scores)
            record = dict(id=sid, image_name=row.get("image_name"),
                          image_info={"image_path": row["image_info"]["image_path"]},
                          question=row.get("question") or row.get("query_question"),
                          qa_info=row.get("qa_info") or {}, target_extraction=row.get("target_extraction") or {},
                          sam_prompts=objects, rgb_sha256=rgb_sha, uses_gt=False)
            saved[sid] = (dataset, scene, record)
            save()
            print(f"[{len(saved)}/{len(rows)}] {sid}", flush=True)
    finally:
        save()


if __name__ == "__main__":
    main()
