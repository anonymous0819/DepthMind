from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from dronepur_common import file_digest, write_json_atomic
from external_tools.depth.extract_dronepur_da3_depth import save_depth
from external_tools.o3dvqa_extraction_common import collect_images
from tcot.cache.vertical_geometry import project


def validate(path, rgb, sha, config):
    with Image.open(rgb) as image:
        shape = (image.height, image.width)
    with np.load(path, allow_pickle=False) as data:
        if data["depth"].shape != shape or data["valid_mask"].shape != shape:
            raise ValueError(f"Unaligned predicted depth: {path}")
        if not np.any(np.isfinite(data["depth"]) & (data["depth"] > 0) & data["valid_mask"].astype(bool)):
            raise ValueError(f"No valid predicted depth: {path}")
        project(data["depth"], data["intrinsics"], np.zeros(shape, dtype=bool))
        if str(data["source_sha256"].item()) != sha or json.loads(str(data["cache_config"].item())) != config:
            raise ValueError(f"Predicted depth source/config differs: {path}")


def run(args, backend, loader, predict, checkpoint, settings, code_path):
    if args.overwrite or args.refresh_conf:
        raise ValueError("Use a new out_root to refresh predicted assets; do not overwrite an existing tool cache")
    root, data_root = Path(args.out_root).resolve(), Path(args.data_dir).resolve()
    if root.is_relative_to(data_root):
        raise ValueError("Depth output must be outside source dataset")
    if args.max_samples < 0:
        raise ValueError("max_samples cannot be negative")
    weight = Path(checkpoint).resolve()
    if not weight.exists():
        raise FileNotFoundError(weight)
    files = [weight] if weight.is_file() else sorted(p for p in weight.rglob("*") if p.is_file())
    identity = {str(p.relative_to(weight)) if weight.is_dir() else p.name:
                dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files}
    config = dict(depth_backend=backend, depth_unit="m", uses_gt=False, checkpoint=str(weight),
                  checkpoint_files=identity, qa_sha256=file_digest(args.qa_file),
                  code_sha256=file_digest(code_path), runner_sha256=file_digest(__file__), settings=settings,
                  max_samples=args.max_samples)
    images = collect_images(args.qa_file, data_root, root)
    if args.max_samples:
        images = dict(list(images.items())[:args.max_samples])
    if not images:
        raise ValueError("No input images")
    manifest_path = root / "manifest.json"
    previous = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["config"] != config:
            raise ValueError("Depth extraction identity changed; use a new out_root")
    entries, pending = {}, []
    for path, rgb in images.items():
        key, sha = path.relative_to(root).as_posix(), file_digest(rgb)
        if path.exists():
            validate(path, rgb, sha, config)
            entry = dict(rgb_sha256=sha, depth_sha256=file_digest(path))
            if key in previous.get("images", {}) and previous["images"][key] != entry:
                raise ValueError(f"Predicted cache fingerprint differs: {path}")
            entries[key] = entry
        else:
            pending.append((path, rgb, key, sha))
    print(f"[{backend}] unique={len(images)} cached={len(entries)} missing={len(pending)}", flush=True)

    def save():
        write_json_atomic(manifest_path, dict(config=config, expected_images=len(images),
                                             complete=len(entries) == len(images), images=entries))

    try:
        save()
        model, device = loader() if pending else (None, None)
        for path, rgb, key, sha in pending:
            payload = {name: value for name, value in predict(model, device, str(rgb)).items() if value is not None}
            save_depth(path, payload, sha, rgb.relative_to(data_root).as_posix(), config)
            validate(path, rgb, sha, config)
            entries[key] = dict(rgb_sha256=sha, depth_sha256=file_digest(path))
            save()
            print(f"[{backend} {len(entries)}/{len(images)}] {key}", flush=True)
    finally:
        save()
