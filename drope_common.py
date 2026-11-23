from __future__ import annotations

import os

import numpy as np

SYSTEM_PROMPT = (
    "You are an assistant who perfectly answer question in urban environment. "
    "Only based on the image, you should directly answer the height, width, volume "
    "and distance question with exact number Answer the distance without output "
    "intermediate process. You should answer the direction question in the direction "
    "of the clock with taking your front as 12 o 'clock, your left as 9 o 'clock, "
    "and your right as 3 o 'clock."
)
DEPTH_NPZ_KEYS = ("depth", "depths", "pred_depth", "pred_depths", "disparity", "disp")
DEPTH_NORM_MODE_LOG = "log"
DEPTH_NORM_MODE_LINEAR = "linear"
DEPTH_NORM_MODE_INVERSE = "inverse"
DEPTH_NORM_MODES = (DEPTH_NORM_MODE_LOG, DEPTH_NORM_MODE_LINEAR, DEPTH_NORM_MODE_INVERSE)


def build_drope_messages(image, question, experiment):
    from train_depth_rope_qwen3vl import build_experiment_prompt

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": build_experiment_prompt(question, experiment)},
        ]},
    ]


def build_data_path(data_dir, dataset, scene, relative_path):
    if relative_path is None:
        return None
    normalized = os.path.normpath(relative_path.replace("\\", os.sep))
    if os.path.isabs(normalized):
        return normalized
    candidates = [os.path.join(data_dir, dataset, scene, normalized), os.path.join(data_dir, normalized), normalized]
    return next((path for path in candidates if os.path.exists(path)), candidates[0])


def build_image_path(data_dir, dataset, scene, image_info):
    path = image_info.get("image_path")
    if path is None:
        raise ValueError("Missing image_info.image_path")
    path = path.replace("\\", os.sep)
    if os.path.basename(path) == path:
        path = os.path.join("rgb", path)
    return build_data_path(data_dir, dataset, scene, path)


def build_depth_path(data_dir, dataset, scene, image_info):
    if image_info.get("depth_path") is not None:
        path = image_info["depth_path"].replace("\\", os.sep)
        if os.path.basename(path) == path:
            path = os.path.join("depth", path)
        return build_data_path(data_dir, dataset, scene, path)
    image_path = image_info.get("image_path")
    if image_path is None:
        return None
    image_path = image_path.replace("\\", "/")
    if os.path.basename(image_path) == image_path:
        image_path = "rgb/" + image_path
    stem = os.path.splitext(image_path.replace("rgb/", "depth/").replace("RGBVis", "Depth"))[0]
    return build_data_path(data_dir, dataset, scene, stem + ".npy")


def build_estimated_depth_path(estimated_depth_root, data_dir, dataset, scene, image_info):
    if not estimated_depth_root or not image_info.get("image_path"):
        return None
    path = image_info["image_path"].replace("\\", os.sep)
    if os.path.basename(path) == path:
        path = os.path.join("rgb", path)
    if os.path.isabs(path):
        path = os.path.relpath(build_image_path(data_dir, dataset, scene, image_info), data_dir)
    stem = os.path.splitext(path)[0]
    basename = os.path.basename(stem)
    stems = [os.path.join(dataset, scene, stem), stem, os.path.join(dataset, scene, basename), basename]
    candidates = []
    for name in stems:
        candidates.extend([name + ".npy", name + ".npz", os.path.join(name, "results.npy"),
                           os.path.join(name, "results.npz"), os.path.join(name, "exports", "mini_npz", "results.npz"),
                           os.path.join(name, "exports", "npz", "results.npz")])
    resolved = [os.path.normpath(os.path.join(estimated_depth_root, candidate)) for candidate in candidates]
    return next((candidate for candidate in resolved if os.path.exists(candidate)), resolved[0])


def load_depth(depth_path):
    loaded = np.load(depth_path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            key = next((key for key in DEPTH_NPZ_KEYS if key in loaded.files), None)
            if key is None:
                raise KeyError(f"No supported depth key in {depth_path}")
            depth = loaded[key]
        finally:
            loaded.close()
    else:
        depth = loaded
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth, got {depth.shape}")
    return depth


def get_valid_depth_mask(depth, invalid_depth_threshold):
    return np.isfinite(depth) & (depth > 0) & (depth < invalid_depth_threshold)


def robust_depth_bounds(depth, valid_mask, lower_percentile, upper_percentile):
    values = depth[valid_mask]
    if not values.size:
        raise ValueError("Depth map has no valid pixels after filtering")
    lower = float(np.percentile(values, lower_percentile))
    upper = float(np.percentile(values, upper_percentile))
    if upper <= lower:
        upper = float(values.max())
    if upper <= lower:
        upper = lower + 1.
    return lower, upper


def normalize_depth(depth, valid_mask, lower, upper, use_log_depth, mode=None):
    mode = mode or (DEPTH_NORM_MODE_LOG if use_log_depth else DEPTH_NORM_MODE_LINEAR)
    if mode not in DEPTH_NORM_MODES:
        raise ValueError(f"Unsupported depth_norm_mode: {mode}")
    safe = np.where(valid_mask, depth, upper).astype(np.float32)
    if mode == DEPTH_NORM_MODE_INVERSE:
        inverse = 1. / np.maximum(safe, 1e-6)
        values = inverse[valid_mask]
        if not values.size:
            raise ValueError("Depth map has no valid pixels for inverse-depth normalization")
        low = float(np.percentile(values, 2.))
        high = float(np.percentile(values, 98.))
        if high <= low:
            high = float(values.max())
        if high <= low:
            high = low + 1e-6
        normalized = np.clip((inverse - low) / (high - low), 0., 1.).astype(np.float32)
        normalized[~valid_mask] = 0.
        return 1. - normalized
    clipped = np.clip(safe, lower, upper)
    if mode == DEPTH_NORM_MODE_LOG:
        log_lower = np.log(max(lower, 1e-6))
        log_upper = np.log(max(upper, lower + 1e-6))
        normalized = (np.log(np.maximum(clipped, 1e-6)) - log_lower) / max(log_upper - log_lower, 1e-6)
    else:
        normalized = (clipped - lower) / max(upper - lower, 1e-6)
    normalized = np.clip(normalized, 0., 1.).astype(np.float32)
    normalized[~valid_mask] = 1.
    return normalized
