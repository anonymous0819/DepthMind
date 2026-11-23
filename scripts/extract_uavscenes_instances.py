#!/usr/bin/env python3
"""Extract benchmark GT instances from official semantic labels and LiDAR.

These GT assets are used only to construct the benchmark, not as model/tool
inputs. Outputs follow input frame order even with multiple workers. A frozen
historical instance order is required to regenerate an existing sampled QA set.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import partial
from pathlib import Path

from build_uavscenes_benchmark import CLASS_CONFIG, GRID_LABELS, filename, load_jsonl


def project(points, rotation, translation, intrinsics, width, height):
    import numpy as np
    camera = (rotation @ points.T).T + translation
    camera = camera[camera[:, 2] > 0.5]
    uv_h = (intrinsics @ camera.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv[valid], camera[valid, 2], camera[valid]


def grid_pos(bbox, width, height):
    col = min(2, int((bbox[0] + bbox[2]) / 2 / (width / 3)))
    row = min(2, int((bbox[1] + bbox[3]) / 2 / (height / 3)))
    return GRID_LABELS[row][col]


def process_frame(frame_and_index, data_root):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import label
    frame, frame_id = frame_and_index
    scene_dir = data_root / f"interval5_{frame['scene']}"
    lidar_path = scene_dir / 'interval5_LIDAR' / filename(frame['lidar_path'])
    label_path = data_root / 'interval5_CAM_label' / f"interval5_{frame['scene']}" / 'interval5_CAM_label_id' / filename(frame['cam_label_path'])
    try:
        points = np.loadtxt(lidar_path, dtype=np.float32)
    except (OSError, ValueError):
        return []
    with Image.open(label_path) as image:
        label_ids = np.asarray(image)
    if label_ids.ndim == 3:
        label_ids = label_ids[..., 0]
    label_ids = label_ids.astype(np.int16)
    width, height = frame['W'], frame['H']
    if label_ids.shape != (height, width):
        return []
    rotation = np.asarray(frame['R'], dtype=np.float64)
    translation = np.asarray(frame['t'], dtype=np.float64)
    intrinsics = np.asarray(frame['K'], dtype=np.float64)
    uv, depth, camera = project(points, rotation, translation, intrinsics, width, height)
    if len(uv) / (height * width) < 0.001:
        return []
    u = np.clip(uv[:, 0].astype(int), 0, width - 1)
    v = np.clip(uv[:, 1].astype(int), 0, height - 1)
    instances = []
    for class_id, config in CLASS_CONFIG.items():
        mask = label_ids == class_id
        if mask.sum() < config['min_area_px']:
            continue
        components, count = label(mask)
        areas = [(int((components == index).sum()), index) for index in range(1, count + 1)]
        areas = sorted((pair for pair in areas if pair[0] >= config['min_area_px']), reverse=True)[:config['top_k']]
        for area, index in areas:
            component = components == index
            ys, xs = np.nonzero(component)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            selected = component[v, u]
            if selected.sum() < 5:
                continue
            cloud = camera[selected]
            extent = cloud[:, :2].max(axis=0) - cloud[:, :2].min(axis=0)
            center = np.median(cloud, axis=0)
            instances.append(dict(
                frame_id=frame_id, scene=frame['scene'], stem=frame['stem'], image_path=filename(frame['image_path']),
                W=width, H=height, class_id=class_id, class_name=config['name'], bbox_px=bbox,
                area_px=area, n_lidar_pts=int(selected.sum()), grid_pos=grid_pos(bbox, width, height),
                depth_median_m=float(np.median(depth[selected])), dist_from_cam_m=float(np.sqrt(sum(float(value) ** 2 for value in center))),
                xyz_center_cam=center.tolist(), horiz_max_m=float(extent.max()), horiz_min_m=float(extent.min()),
                supports_size=config['supports_size'], supports_distance=config['supports_distance'], supports_pair=config['supports_pair'],
            ))
    class_counts = Counter(instance['class_name'] for instance in instances)
    grid_counts = Counter((instance['class_name'], instance['grid_pos']) for instance in instances)
    for instance in instances:
        count = class_counts[instance['class_name']]
        grid_count = grid_counts[(instance['class_name'], instance['grid_pos'])]
        instance.update(n_same_class_in_frame=count, n_same_class_in_grid=grid_count,
                        is_unique_in_frame=count == 1, is_unique_in_grid=grid_count == 1)
    return instances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--frame-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    frames = load_jsonl(args.frame_index)
    work = [(frame, index) for index, frame in enumerate(frames)]
    worker = partial(process_frame, data_root=args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from multiprocessing import Pool
    with args.output.open('w', encoding='utf-8') as handle, Pool(args.workers) as pool:
        for instances in pool.imap(worker, work, chunksize=4):
            for instance in instances:
                handle.write(json.dumps(instance) + '\n')
    print(f'Saved instances: {args.output}')


if __name__ == '__main__':
    main()
