#!/usr/bin/env python3
"""Index original UAVScenes RGB, LiDAR, semantic labels and calibration.

The output uses filenames only. Raw assets remain in the official UAVScenes
layout and are located at runtime using --data-root and each record's scene.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from build_uavscenes_benchmark import TEST_SCENES, filename


def build_scene_index(data_root, scene, calibrations):
    scene_dir = data_root / f'interval5_{scene}'
    infos = json.loads((scene_dir / 'sampleinfos_interpolated.json').read_text(encoding='utf-8'))
    lidar_dir = scene_dir / 'interval5_LIDAR'
    lidar_names = {}
    for path in sorted(lidar_dir.iterdir()):
        if path.suffix == '.txt':
            stem = path.name[len('image'):].split('_lidar')[0]
            if stem in lidar_names:
                raise ValueError(f'Duplicate LiDAR frame: {scene}/{stem}')
            lidar_names[stem] = path.name
    label_dir = data_root / 'interval5_CAM_label' / f'interval5_{scene}' / 'interval5_CAM_label_id'
    calibration = calibrations[scene]
    r = calibration['camera_ext_R']
    t = calibration['camera_ext_t']
    import numpy as np
    rotation = np.asarray(r, dtype=float).reshape(3, 3).tolist()
    translation = np.asarray(t, dtype=float).reshape(3).tolist()
    entries = []
    for info in infos:
        image_name = filename(info['OriginalImageName'])
        stem = image_name.rsplit('.', 1)[0]
        lidar_name = lidar_names.get(stem)
        label_name = stem + '.png'
        if lidar_name is None or not (label_dir / label_name).is_file() or not (scene_dir / 'interval5_CAM' / image_name).is_file():
            continue
        entries.append(dict(scene=scene, stem=stem, image_path=image_name, lidar_path=lidar_name,
                            cam_label_path=label_name, K=info['P3x3'], R=rotation, t=translation,
                            W=int(info['Width']), H=int(info['Height'])))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, required=True,
                        help='Official UAVScenes repository directory containing calibration_results.py.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.calibration_dir.resolve()))
    from calibration_results import scenename_to_calibration
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        for scene in TEST_SCENES:
            entries = build_scene_index(args.data_root, scene, scenename_to_calibration)
            for entry in entries:
                handle.write(json.dumps(entry) + '\n')
            print(f'{scene}: {len(entries)} frames')


if __name__ == '__main__':
    main()
