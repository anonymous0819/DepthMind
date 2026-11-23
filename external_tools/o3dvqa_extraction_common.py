from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def iter_items(data, dataset="", scene=""):
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError("Every extraction input must have a sample ID")
            yield dataset, scene, row
    elif isinstance(data, dict):
        for key, value in data.items():
            if not dataset:
                yield from iter_items(value, str(key), scene)
            else:
                yield from iter_items(value, dataset, "/".join(filter(None, (scene, str(key)))))
    else:
        raise ValueError("Expected nested dataset/scene/items JSON")


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(str(relative).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a relative asset path: {relative}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Asset path escapes its root: {relative}")
    return target


def build_image_path(data_dir, dataset, scene, image_info):
    relative = Path(image_info["image_path"].replace("\\", "/"))
    if len(relative.parts) == 1:
        relative = Path("rgb") / relative
    path = safe_path(data_dir, Path(dataset) / scene / relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def depth_path(root, dataset, scene, image_info):
    relative = Path(image_info["image_path"].replace("\\", "/"))
    if len(relative.parts) == 1:
        relative = Path("rgb") / relative
    return safe_path(root, Path(dataset) / scene / relative.with_suffix(".npz"))


def collect_images(qa_file, data_dir, out_root):
    images, ids = {}, set()
    data = json.loads(Path(qa_file).read_text(encoding="utf-8"))
    for dataset, scene, row in iter_items(data):
        sid = str(row["id"])
        if sid in ids:
            raise ValueError(f"Duplicate sample ID: {sid}")
        ids.add(sid)
        info = row["image_info"]
        rgb = Path(build_image_path(data_dir, dataset, scene, info)).resolve()
        target = depth_path(out_root, dataset, scene, info)
        if target in images and images[target] != rgb:
            raise ValueError(f"Different images collide at {target}")
        images[target] = rgb
    return dict(sorted(images.items()))


def normalize_relpath(path):
    return str(path).replace("\\", "/")


def resize_depth_to(depth, target_hw):
    if depth.shape == tuple(target_hw):
        return depth
    h, w = target_hw
    return np.asarray(Image.fromarray(depth.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR))


def resize_mask_to(mask, target_hw):
    if mask.shape == tuple(target_hw):
        return mask.astype(bool)
    h, w = target_hw
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)) > 0


def load_mask_from_npz(path):
    if not path:
        return None
    with np.load(path, allow_pickle=False) as data:
        if "masks" not in data and "mask" not in data:
            raise ValueError(f"Missing predicted mask array: {path}")
        mask = data["masks" if "masks" in data else "mask"].astype(bool)
    if mask.ndim == 3:
        mask = np.any(mask, axis=0)
    if mask.ndim != 2:
        raise ValueError(f"Predicted mask must be 2D or stacked 2D: {path}")
    return mask


def load_sam3_index(path):
    index = {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for _, _, row in iter_items(payload):
        sid = str(row["id"])
        if sid in index:
            raise ValueError(f"Duplicate mask ID: {sid}")
        entry = {}
        for role, obj in (row.get("sam_prompts") or {}).items():
            if obj.get("num_masks", 0) and obj.get("mask_path"):
                asset = Path(obj["mask_path"])
                if not asset.is_absolute():
                    asset = Path(path).resolve().parent / asset
                entry[role] = str(asset.resolve())
        if entry:
            index[sid] = entry
    return index


def bind_output_identity(path, config, resume):
    from dronepur_common import write_json_atomic

    path = Path(path).resolve()
    manifest = path.with_name(path.name + ".identity.json")
    if path.exists() and not resume:
        raise FileExistsError("Output exists; use --resume or a new output")
    if path.exists() and not manifest.exists():
        raise ValueError("Existing extraction has no verified identity; use a new output")
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != config:
        raise ValueError("Extraction inputs/configuration changed; use a new output")
    write_json_atomic(manifest, config)
