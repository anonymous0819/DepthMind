#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from dronepur_common import file_digest, write_json_atomic
from external_tools.o3dvqa_extraction_common import build_image_path, depth_path, iter_items, load_mask_from_npz
from tcot.cache.vertical_geometry import project


def build_manifest(mask_file, data_dir, depth_root, depth_unit, depth_backend, output):
    if depth_backend == "da3" and depth_unit != "predicted_depth_units":
        raise ValueError("DA3 extraction must not be relabeled as metric meters")
    payload = json.loads(Path(mask_file).read_text(encoding="utf-8"))
    entries = {}
    for dataset, scene, row in iter_items(payload):
        sid = str(row["id"])
        if sid in entries:
            raise ValueError(f"Duplicate perception ID: {sid}")
        if row.get("uses_gt") is True:
            raise ValueError(f"GT masks are not permitted: {sid}")
        rgb = Path(build_image_path(data_dir, dataset, scene, row["image_info"]))
        rgb_sha = file_digest(rgb)
        if row.get("rgb_sha256") and rgb_sha != row["rgb_sha256"]:
            raise ValueError(f"Source image differs from mask extraction: {sid}")
        stem = Path(row["image_info"]["image_path"].replace("\\", "/")).stem
        path = Path(depth_root) / dataset / scene / "rgb" / f"{stem}.npz"
        metadata_present = False
        with np.load(path, allow_pickle=False) as data:
            if "cache_config" in data:
                metadata_present = True
                settings = json.loads(str(data["cache_config"].item()))
                if (not isinstance(settings, dict) or settings.get("uses_gt") is True
                        or settings.get("depth_backend") and settings["depth_backend"] != depth_backend
                        or settings.get("depth_unit") and settings["depth_unit"] != depth_unit):
                    raise ValueError(f"Actual depth backend/unit differs from requested manifest: {path}")
            depth = data["depth"]
            k = data["K"] if "K" in data else data["intrinsics"]
            if depth.ndim != 2:
                raise ValueError(f"Depth must be 2D: {path}")
            if "valid_mask" in data and data["valid_mask"].shape != depth.shape:
                raise ValueError(f"Depth validity shape differs: {path}")
            project(depth, k, np.zeros(depth.shape, dtype=bool))
            if min(k[0, 0], k[1, 1]) < 2:
                raise ValueError(f"Intrinsics must be in pixels: {path}")
            if "source_sha256" in data and str(data["source_sha256"].item()) != rgb_sha:
                raise ValueError(f"Depth belongs to a different image: {path}")
        objects, fingerprints = {}, {}
        for role, obj in row["sam_prompts"].items():
            if role not in {"subject", "reference"}:
                raise ValueError(f"Unexpected object role: {role}")
            if not obj["num_masks"]:
                continue
            mask_path = Path(obj["mask_path"])
            if not mask_path.is_absolute():
                mask_path = Path(mask_file).resolve().parent / mask_path
            mask_sha = file_digest(mask_path)
            if obj.get("mask_sha256") and mask_sha != obj["mask_sha256"]:
                raise ValueError(f"Predicted mask differs from index: {sid}/{role}")
            load_mask_from_npz(mask_path)
            objects[role] = dict(prompt=obj["prompt"], mask_path=os.path.relpath(mask_path, Path(output).resolve().parent))
            fingerprints[role] = mask_sha
        entries[sid] = dict(depth_path=os.path.relpath(path, Path(output).resolve().parent),
                            depth_unit=depth_unit, objects=objects, uses_gt=False,
                            image_name=row.get("image_name"), question=row["question"],
                            asset_status="ok" if "subject" in objects and len(objects) == len(row["sam_prompts"]) else "missing_predicted_masks",
                            source=dict(depth_backend=depth_backend, rgb_sha256=rgb_sha,
                                        provenance="extraction_metadata_present" if metadata_present else "historical_backend_declared",
                                        depth_sha256=file_digest(path), mask_sha256=fingerprints))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Build ID-to-predicted-assets manifest for DepthMind geometric tools; no scoring")
    parser.add_argument("--mask_file", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--depth_root", type=Path, required=True)
    parser.add_argument("--depth_backend", choices=("da3", "moge2", "unidepth"), required=True)
    parser.add_argument("--depth_unit", choices=("m", "predicted_depth_units"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.mask_file.resolve() or args.output.exists():
        parser.error("Use a new output path; do not overwrite input or existing manifests")
    if args.output.resolve().is_relative_to(args.data_dir.resolve()):
        parser.error("Output must be outside the source dataset")
    entries = build_manifest(args.mask_file, args.data_dir, args.depth_root, args.depth_unit, args.depth_backend, args.output)
    write_json_atomic(args.output, entries)
    summary = dict(uses_gt=False, samples=len(entries),
                   available=sum(row["asset_status"] == "ok" for row in entries.values()),
                   mask_index_sha256=file_digest(args.mask_file), output_sha256=file_digest(args.output),
                   code_sha256=file_digest(__file__), depth_backend=args.depth_backend, depth_unit=args.depth_unit)
    write_json_atomic(args.output.with_name(args.output.name + ".manifest.json"), summary)
    print(f"Perception assets: {summary['available']}/{summary['samples']} available; {args.output}", flush=True)


if __name__ == "__main__":
    main()
