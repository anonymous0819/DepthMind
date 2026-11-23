#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dronepur_common import file_digest, write_json_atomic
from external_tools.depth.extract_dronepur_da3_depth import load_da3
from external_tools.o3dvqa_extraction_common import build_image_path, iter_items, safe_path

FIELDS = {"depth", "source_image", "conf", "intrinsics", "extrinsics"}
PATCH_SIZE = 14


def native_shape(image_size, process_res):
    width, height = image_size
    scale = process_res / float(max(width, height))

    def patch_multiple(value):
        scaled = max(1, int(round(value * scale)))
        down = (scaled // PATCH_SIZE) * PATCH_SIZE
        up = down + PATCH_SIZE
        return max(1, up if abs(up - scaled) <= abs(scaled - down) else down)

    return patch_multiple(height), patch_multiple(width)


def collect_images(qa_file, data_dir, out_root):
    images, ids = {}, set()
    payload = json.loads(Path(qa_file).read_text(encoding="utf-8"))
    for dataset, scene, row in iter_items(payload):
        sid = str(row["id"])
        if sid in ids:
            raise ValueError(f"Duplicate sample ID: {sid}")
        ids.add(sid)
        rgb = Path(build_image_path(data_dir, dataset, scene, row["image_info"])).resolve()
        stem = Path(row["image_info"]["image_path"].replace("\\", "/")).stem
        path = safe_path(out_root, Path(dataset) / scene / "rgb" / f"{stem}.npz")
        if path in images and images[path] != rgb:
            raise ValueError(f"Different RGB images collide at {path}")
        images[path] = rgb
    return dict(sorted(images.items()))


def predict_depth(model, rgb_path, process_res):
    import numpy as np
    import torch

    with torch.inference_mode():
        prediction = model.inference(
            [str(rgb_path)], process_res=process_res, process_res_method="upper_bound_resize",
        )
    if prediction.conf is None or prediction.intrinsics is None or prediction.extrinsics is None:
        raise ValueError("Historical native cache requires DA3 confidence, intrinsics and extrinsics")
    return dict(depth=np.asarray(prediction.depth[0], dtype=np.float32),
                conf=np.asarray(prediction.conf[0], dtype=np.float32),
                intrinsics=np.asarray(prediction.intrinsics[0], dtype=np.float32),
                extrinsics=np.asarray(prediction.extrinsics[0], dtype=np.float32))


def save_native(path, arrays, source_image):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".da3-native-", suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, depth=arrays["depth"], source_image=np.array(source_image),
                                conf=arrays["conf"], intrinsics=arrays["intrinsics"],
                                extrinsics=arrays["extrinsics"])
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_native(path, rgb_path, relative, process_res):
    import numpy as np
    from PIL import Image

    with Image.open(rgb_path) as image:
        image_size = list(image.size)
    expected = native_shape(image_size, process_res)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != FIELDS:
            raise ValueError(f"Expected historical native DA3 fields; use a separate output for other formats: {path}")
        depth, conf = archive["depth"], archive["conf"]
        if depth.shape != expected or conf.shape != expected:
            raise ValueError(f"DA3 native canvas must be {expected}, not original-resolution depth: {path}")
        if depth.dtype != np.float32 or conf.dtype != np.float32:
            raise ValueError(f"Expected float32 native DA3 depth/confidence: {path}")
        if not np.any(np.isfinite(depth) & (depth > 0) & (depth < 65000.)) or not np.isfinite(conf).all():
            raise ValueError(f"Invalid DA3 depth/confidence: {path}")
        k, ext = archive["intrinsics"], archive["extrinsics"]
        if (k.shape != (3, 3) or ext.shape != (3, 4) or k.dtype != np.float32 or ext.dtype != np.float32
                or not np.isfinite(k).all() or not np.isfinite(ext).all()
                or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])):
            raise ValueError(f"Expected native pixel K and unpadded 3x4 extrinsics: {path}")
        source = archive["source_image"]
        if source.shape != () or source.dtype.kind not in "US" or str(source.item()).replace("\\", "/") != relative:
            raise ValueError(f"Cached source_image does not match the RGB path: {path}")
    return dict(image_size=image_size, depth_shape=list(expected), source_image=relative)


def main():
    parser = argparse.ArgumentParser(description="Extract historical-format native DA3 depth/conf/K/3x4 extrinsics from Open3DVQA RGB")
    parser.add_argument("--qa_file", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--da3_src", type=Path, required=True)
    parser.add_argument("--da3_model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    if args.process_res < PATCH_SIZE or args.process_res % PATCH_SIZE:
        parser.error("process_res must be a positive multiple of the DA3 patch size (14)")
    out_root, data_root = args.out_root.resolve(), args.data_dir.resolve()
    if out_root.is_relative_to(data_root):
        parser.error("Depth output must be outside the source dataset")
    images = collect_images(args.qa_file, data_root, out_root)
    if not images:
        raise ValueError("No input images")
    config = dict(da3_model=str(args.da3_model.resolve()), process_res=args.process_res,
                  process_res_method="upper_bound_resize", patch_size=PATCH_SIZE,
                  depth_format="native_processed_resolution", fields=sorted(FIELDS),
                  extraction_sha256=file_digest(__file__), loader_sha256=file_digest(load_da3.__code__.co_filename),
                  qa_sha256=file_digest(args.qa_file), uses_gt=False, depth_backend="da3",
                  depth_unit="model_native_scale_no_conversion")
    manifest_path = out_root / "manifest.json"
    previous = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config") != config:
            raise ValueError("Extraction identity/format changed; use a new out_root")
    entries, pending = {}, []
    for path, rgb in images.items():
        key, relative = path.relative_to(out_root).as_posix(), rgb.relative_to(data_root).as_posix()
        sha = file_digest(rgb)
        if path.exists():
            if not args.skip_existing and not args.check_only:
                raise FileExistsError(f"Use --skip_existing: {path}")
            entry = dict(validate_native(path, rgb, relative, args.process_res),
                         rgb_sha256=sha, depth_sha256=file_digest(path))
            expected = previous.get("images", {}).get(key)
            if expected is not None:
                if any(entry[name] != expected.get(name) for name in entry):
                    raise ValueError(f"Cached depth/source differs from manifest: {path}")
                entry["origin"] = expected.get("origin", "generated_native")
            else:
                entry["origin"] = "existing_native_cache_settings_unverified"
            entries[key] = entry
        else:
            pending.append((path, rgb, key, relative, sha))
    print(f"Native DA3 images: {len(images)}; valid cached: {len(entries)}; missing: {len(pending)}", flush=True)
    if args.check_only:
        if pending:
            raise FileNotFoundError(f"Native DA3 cache is incomplete: {len(pending)} images missing")
        return

    def save_manifest():
        write_json_atomic(manifest_path, dict(config=config, expected_images=len(images),
                                             complete=len(entries) == len(images), images=entries))

    try:
        save_manifest()
        model = load_da3(args.da3_src, str(args.da3_model), args.device) if pending else None
        for index, (path, rgb, key, relative, sha) in enumerate(pending, 1):
            arrays = predict_depth(model, rgb, args.process_res)
            save_native(path, arrays, relative)
            entries[key] = dict(validate_native(path, rgb, relative, args.process_res),
                                rgb_sha256=sha, depth_sha256=file_digest(path), origin="generated_native")
            print(f"[{len(entries)}/{len(images)}] {key}", flush=True)
            if index % 10 == 0:
                save_manifest()
    finally:
        save_manifest()


if __name__ == "__main__":
    main()
