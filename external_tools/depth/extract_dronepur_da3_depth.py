#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("USE_TF", "0")

from dronepur_common import DEPTH_IMAGE_COUNTS, TASKS, cache_path, collect_images, file_digest, validate_depth, write_json_atomic


def load_da3(source_dir, model_path, device):
    import torch

    sys.path.insert(0, str(Path(source_dir).resolve()))
    from depth_anything_3.api import DepthAnything3

    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"Expected local DA3 weights: {model_path}")
    model = DepthAnything3.from_pretrained(model_path).to(torch.device(device)).eval()
    model.device = torch.device(device)
    return model


def predict_depth(model, image, process_res):
    import numpy as np
    import torch
    from PIL import Image

    with torch.inference_mode():
        prediction = model.inference([image], process_res=process_res, process_res_method="upper_bound_resize")
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected DA3 (H,W) depth, got {depth.shape}")
    processed_h, processed_w = depth.shape
    width, height = image.size
    k = np.asarray(prediction.intrinsics[0], dtype=np.float32).reshape(3, 3).copy()
    k[0, :] *= width / processed_w
    k[1, :] *= height / processed_h
    with Image.fromarray(depth, mode="F") as plane:
        depth = np.asarray(plane.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32).copy()
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("DA3 returned invalid pixel intrinsics")
    if not np.any(np.isfinite(depth) & (depth > 0) & (depth < 65000.)):
        raise ValueError("DA3 depth has no valid pixels")
    return dict(depth=depth, intrinsics=k)


def save_depth(path, arrays, source_sha256, relative, config):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".da3-", suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays, source_sha256=source_sha256,
                                source_relative=relative, cache_config=json.dumps(config, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def reuse_index(depth_roots, target_root, config):
    sources, images = {}, {}
    for value in depth_roots:
        root = Path(value).resolve()
        key = str(root)
        if root == target_root or key in sources:
            raise ValueError("Reuse roots must be distinct from each other and the destination")
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError(f"Reuse cache extraction is incomplete: {root}")
        settings = manifest["config"]
        if (Path(settings["da3_model"]).resolve() != Path(config["da3_model"]).resolve()
                or any(settings[name] != config[name] for name in
                       ("process_res", "process_res_method", "depth_format"))):
            raise ValueError(f"Reuse cache DA3 model/resolution/format differs: {root}")
        sources[key] = dict(manifest_sha256=file_digest(manifest_path), config=settings)
        for relative, entry in manifest["images"].items():
            images.setdefault(relative, (cache_path(root, relative), dict(
                entry, cache_config=entry.get("cache_config", settings), reused_from=key)))
    return sources, images


def copy_depth(source, destination, expected_sha256):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".da3-reuse-", suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as output, Path(source).open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
        if file_digest(temporary) != expected_sha256:
            raise ValueError(f"Reused depth changed during copying: {source}")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description="Extract per-image raw DA3 depth from original Drone-PUR images, without reference-box overlays")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=tuple(DEPTH_IMAGE_COUNTS), default=list(TASKS))
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--reuse-depth-root", type=Path, nargs="+", action="extend", default=[],
                        help="Copy validated existing depths from these roots; first matching root wins")
    parser.add_argument("--da3-src", type=Path, required=True, help="Depth-Anything-3 source directory containing depth_anything_3")
    parser.add_argument("--da3-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--skip-existing", action="store_true", help="Validate and reuse compatible per-image caches")
    parser.add_argument("--check-only", action="store_true", help="Count unique inputs and check available caches without loading DA3")
    args = parser.parse_args()
    if args.process_res < 1:
        parser.error("process-res must be positive")
    data_root, depth_root = args.data_root.resolve(), args.depth_root.resolve()
    if depth_root == data_root or any(depth_root.is_relative_to(data_root / part) for part in ("images", "annotations", "metadata")):
        parser.error("Depth cache must not overwrite benchmark assets")
    images, counts = collect_images(data_root, args.tasks)
    print(f"Questions: {counts}; unique RGB images: {len(images)}", flush=True)
    destinations = [cache_path(depth_root, relative) for relative in images]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Different image extensions collide at the same NPZ path")
    config = dict(da3_model=str(args.da3_model.resolve()), process_res=args.process_res,
                  process_res_method="upper_bound_resize", depth_format="raw_original_resolution",
                  extraction_sha256=file_digest(__file__))
    reuse_sources, reusable = reuse_index(args.reuse_depth_root, depth_root, config)
    manifest_path = depth_root / "manifest.json"
    previous = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (previous.get("config") != config or previous.get("tasks") != counts
                or previous.get("reuse_sources", {}) != reuse_sources):
            raise ValueError("Depth manifest settings/tasks/reuse sources differ; use a new depth root")
    entries, pending, copies = {}, [], []
    for relative, rgb in images.items():
        path = cache_path(depth_root, relative)
        source_sha = file_digest(rgb)
        candidate = reusable.get(relative)
        expected = previous.get("images", {}).get(relative)
        if expected is None and candidate is not None:
            expected = candidate[1]
        if path.exists():
            if not args.skip_existing and not args.check_only:
                raise FileExistsError(f"Cache exists; use --skip-existing or a new depth root: {path}")
            entry_config = expected.get("cache_config", config) if expected is not None else config
            size = validate_depth(path, rgb, source_sha, entry_config)
            entry = dict(rgb_sha256=source_sha, depth_sha256=file_digest(path), image_size=size)
            if expected is not None:
                if any(entry[name] != expected[name] for name in entry):
                    raise ValueError(f"Cached RGB/depth differs from its manifest: {relative}")
                entry = dict(expected)
            entries[relative] = entry
        elif candidate is not None:
            source, entry = candidate
            if source_sha != entry["rgb_sha256"] or file_digest(source) != entry["depth_sha256"]:
                raise ValueError(f"Reuse RGB/depth changed since extraction: {relative}")
            if validate_depth(source, rgb, source_sha, entry["cache_config"]) != entry["image_size"]:
                raise ValueError(f"Reuse depth image size differs: {relative}")
            copies.append((relative, source, path, entry))
        else:
            pending.append((relative, rgb, path, source_sha))
    print(f"Valid local caches: {len(entries)}; reusable: {len(copies)}; "
          f"DA3 missing: {len(pending)}", flush=True)
    if args.check_only:
        return

    def save_manifest():
        write_json_atomic(manifest_path, dict(config=config, tasks=counts, expected_images=len(images),
                                             reuse_sources=reuse_sources,
                                             complete=len(entries) == len(images), images=dict(sorted(entries.items()))))

    try:
        save_manifest()
        for index, (relative, source, path, entry) in enumerate(copies, 1):
            copy_depth(source, path, entry["depth_sha256"])
            entries[relative] = entry
            print(f"[reuse {index}/{len(copies)}] {relative}", flush=True)
            if index % 10 == 0:
                save_manifest()
        save_manifest()
        model = load_da3(args.da3_src, str(args.da3_model), args.device) if pending else None
        from PIL import Image

        for index, (relative, rgb, path, source_sha) in enumerate(pending, 1):
            with Image.open(rgb) as handle:
                image = handle.convert("RGB")
            try:
                arrays = predict_depth(model, image, args.process_res)
                save_depth(path, arrays, source_sha, relative, config)
            finally:
                image.close()
            size = validate_depth(path, rgb, source_sha, config)
            entries[relative] = dict(rgb_sha256=source_sha, depth_sha256=file_digest(path), image_size=size)
            print(f"[{len(entries)}/{len(images)}] {relative}", flush=True)
            if index % 10 == 0:
                save_manifest()
    finally:
        save_manifest()
    print(f"Saved {len(entries)} depth caches and {manifest_path}")


if __name__ == "__main__":
    main()
