#!/usr/bin/env python3
"""Build the fixed-bin UAVScenes-Metric v2 benchmark from ordered instances.

Inputs are JSONL instances and frame-index records produced by the companion
scripts. Preserve their order to reproduce a particular sampled question set.
Use --v1-qa to retain an existing v1 question set while updating its size GT.
Intervals are strictly open; references on a boundary are rejected rather than
assigned a label. Published questions contain no boundary references, and their
historical option text and answer letters are retained unchanged.
Images are referenced by filename; scene identifies their source directory.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

TEST_SCENES = ['AMtown01', 'HKairport01', 'HKairport02', 'HKairport03', 'HKisland01', 'HKisland02']
SUBSET_GROUPS = {
    'airport': ['HKairport01', 'HKairport02', 'HKairport03'],
    'island': ['HKisland01', 'HKisland02'],
    'town': ['AMtown01'],
}
BINS = {
    'abs_distance_ego': ([0, 50, 80, 110, math.inf], ['<50 m', '50-80 m', '80-110 m', '>110 m']),
    'abs_size_horiz': ([0, 5, 10, 20, math.inf], ['<5 m', '5-10 m', '10-20 m', '>20 m']),
    'abs_distance_obj': ([0, 20, 50, 100, math.inf], ['<20 m', '20-50 m', '50-100 m', '>100 m']),
}
TASK_QUOTA = {'abs_distance_ego': 800, 'abs_size_horiz': 700, 'abs_distance_obj': 500}
TEMPLATES = {
    'abs_distance_ego': [
        'How far is {ref} from the drone (camera)?',
        'What is the approximate distance from the drone to {ref}?',
        'Roughly how far away is {ref} from the camera?',
    ],
    'abs_size_horiz': [
        'What is the approximate horizontal size (largest dimension) of {ref}?',
        'How large is {ref} horizontally?',
        'Estimate the horizontal extent of {ref}.',
    ],
    'abs_distance_obj': [
        'What is the approximate distance between {refA} and {refB}?',
        'How far apart are {refA} and {refB}?',
        'Estimate the distance between {refA} and {refB}.',
    ],
}
CLASS_CONFIG = {
    20: dict(name='sedan', min_area_px=200, top_k=20, supports_size=True, supports_distance=True, supports_pair=True),
    9: dict(name='container', min_area_px=400, top_k=15, supports_size=True, supports_distance=True, supports_pair=True),
    24: dict(name='truck', min_area_px=400, top_k=15, supports_size=True, supports_distance=True, supports_pair=True),
    1: dict(name='roof', min_area_px=600, top_k=15, supports_size=True, supports_distance=True, supports_pair=True),
    10: dict(name='airstrip', min_area_px=2000, top_k=5, supports_size=True, supports_distance=True, supports_pair=True),
    4: dict(name='river', min_area_px=2000, top_k=5, supports_size=True, supports_distance=True, supports_pair=True),
    18: dict(name='car_park', min_area_px=2000, top_k=5, supports_size=True, supports_distance=True, supports_pair=True),
    11: dict(name='traffic_barrier', min_area_px=800, top_k=10, supports_size=False, supports_distance=True, supports_pair=True),
}
GRID_LABELS = [['top-left', 'top', 'top-right'], ['middle-left', 'center', 'middle-right'], ['bottom-left', 'bottom', 'bottom-right']]


def filename(path):
    return str(path).replace('\\', '/').rsplit('/', 1)[-1]


def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_reference(instance):
    name = instance['class_name'].replace('_', ' ')
    if instance['is_unique_in_frame']:
        return f'the {name}'
    if instance['is_unique_in_grid']:
        return f"the {name} in the {instance['grid_pos']} region of the image"
    return None


def bin_answer(value, task):
    edges, labels = BINS[task]
    if not math.isfinite(value):
        raise ValueError(f'Non-finite reference for {task}: {value}')
    for index, label in enumerate(labels):
        if edges[index] < value < edges[index + 1]:
            return label, index
    raise ValueError(f'Reference on an interval boundary or outside the positive domain for {task}: {value}')


def scene_to_subset(scene):
    return next((name for name, scenes in SUBSET_GROUPS.items() if scene in scenes), 'unknown')


def build_single(instances, task):
    candidates = []
    for instance in instances:
        if not instance['supports_size' if task == 'abs_size_horiz' else 'supports_distance']:
            continue
        reference = make_reference(instance)
        if reference is None or (task == 'abs_size_horiz' and instance['n_lidar_pts'] < 15):
            continue
        value = instance['horiz_max_m' if task == 'abs_size_horiz' else 'dist_from_cam_m']
        label, index = bin_answer(value, task)
        candidates.append(dict(
            task=task, frame_id=instance['frame_id'], scene=instance['scene'],
            subset=scene_to_subset(instance['scene']), image_path=filename(instance['image_path']),
            gt_value_m=value, gt_bin=label, gt_bin_idx=index, ref=reference,
            class_name=instance['class_name'], bbox_px=instance['bbox_px'], grid_pos=instance['grid_pos'],
        ))
    return candidates


def build_pairs(instances, rng, per_frame_cap=3):
    by_frame = defaultdict(list)
    for instance in instances:
        by_frame[instance['frame_id']].append(instance)
    candidates = []
    for frame_id, entries in by_frame.items():
        pool = [entry for entry in entries if entry['supports_pair'] and make_reference(entry) is not None]
        pairs = []
        for index, a in enumerate(pool):
            for b in pool[index + 1:]:
                ref_a, ref_b = make_reference(a), make_reference(b)
                if ref_a == ref_b:
                    continue
                value = math.sqrt(sum((a['xyz_center_cam'][axis] - b['xyz_center_cam'][axis]) ** 2 for axis in range(3)))
                if value < 2.0:
                    continue
                label, bin_index = bin_answer(value, 'abs_distance_obj')
                pairs.append((a, b, ref_a, ref_b, value, label, bin_index))
        rng.shuffle(pairs)
        for a, b, ref_a, ref_b, value, label, bin_index in pairs[:per_frame_cap]:
            candidates.append(dict(
                task='abs_distance_obj', frame_id=frame_id, scene=a['scene'], subset=scene_to_subset(a['scene']),
                image_path=filename(a['image_path']), gt_value_m=value, gt_bin=label, gt_bin_idx=bin_index,
                refA=ref_a, refB=ref_b, classA=a['class_name'], classB=b['class_name'],
                bboxA_px=a['bbox_px'], bboxB_px=b['bbox_px'],
            ))
    return candidates


def balance_sample(candidates, quota, rng):
    if len(candidates) <= quota:
        rng.shuffle(candidates)
        return candidates
    buckets = defaultdict(list)
    for candidate in candidates:
        buckets[(candidate['subset'], candidate['gt_bin_idx'])].append(candidate)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    picked = []
    while len(picked) < quota:
        added = 0
        for bucket in buckets.values():
            if bucket:
                picked.append(bucket.pop())
                added += 1
                if len(picked) >= quota:
                    break
        if not added:
            break
    return picked


def render_qa(candidate, index, rng):
    task = candidate['task']
    template = rng.choice(TEMPLATES[task])
    question = template.format(refA=candidate['refA'], refB=candidate['refB']) if task == 'abs_distance_obj' else template.format(ref=candidate['ref'])
    choices = BINS[task][1][:]
    letter = 'ABCD'[candidate['gt_bin_idx']]
    full_question = question + '\n' + '\n'.join(f'({label}) {choice}' for label, choice in zip('ABCD', choices)) + '\nAnswer with the letter of the correct option in \\boxed{}.'
    excluded = {'task', 'subset', 'scene', 'image_path', 'gt_value_m', 'gt_bin', 'gt_bin_idx'}
    return dict(
        id=f'uavmetric-{task}-{index:05d}', task=task, subset=candidate['subset'], scene=candidate['scene'],
        image_path=filename(candidate['image_path']), question=full_question, question_short=question,
        choices=choices, answer_letter=letter, answer_label=choices[candidate['gt_bin_idx']],
        gt_value_m=candidate['gt_value_m'], gt_bin=candidate['gt_bin'],
        meta={key: value for key, value in candidate.items() if key not in excluded},
    )


def build_questions(instances, seed=42):
    rng = random.Random(seed)
    groups = [build_single(instances, 'abs_distance_ego'), build_single(instances, 'abs_size_horiz'), build_pairs(instances, rng)]
    selected = []
    for task, candidates in zip(TASK_QUOTA, groups):
        picked = balance_sample(candidates, TASK_QUOTA[task], rng)
        if len(picked) != TASK_QUOTA[task]:
            raise ValueError(f'Insufficient candidates for {task}: {len(picked)}')
        selected.extend(picked)
    return [render_qa(candidate, index, rng) for index, candidate in enumerate(selected)]


def update_size_gt(questions, instances, frames):
    lookup = {(entry['scene'], entry['stem'], entry['class_name'], tuple(entry['bbox_px'])): entry for entry in instances}
    calibration = {(frame['scene'], frame['stem']): frame['K'] for frame in frames}
    for question in questions:
        question['image_path'] = filename(question['image_path'])
        if question['task'] != 'abs_size_horiz':
            continue
        meta = question['meta']
        stem = question['image_path'][:-4] if question['image_path'].endswith('.jpg') else question['image_path']
        instance = lookup.get((question['scene'], stem, meta['class_name'], tuple(meta['bbox_px'])))
        k = calibration.get((question['scene'], stem))
        if instance is None or k is None or instance.get('depth_median_m', 0) <= 0.5:
            raise ValueError(f"Missing v2 size calibration/instance for {question['id']}")
        fx = float(k[0][0])
        if not math.isfinite(fx) or fx <= 0:
            raise ValueError(f"Invalid focal length for {question['id']}")
        x0, y0, x1, y1 = meta['bbox_px']
        value = max(x1 - x0, y1 - y0) * instance['depth_median_m'] / fx
        label, index = bin_answer(value, 'abs_size_horiz')
        question.update(gt_value_m=float(value), gt_bin=label, answer_letter='ABCD'[index], answer_label=label)
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instances', type=Path, required=True)
    parser.add_argument('--frame-index', type=Path, required=True)
    parser.add_argument('--v1-qa', type=Path, help='Optional frozen v1 QA; only update its size GT, without resampling.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'benchmarks/uavscenes_metric_v2.json')
    args = parser.parse_args()
    instances = load_jsonl(args.instances)
    questions = json.loads(args.v1_qa.read_text(encoding='utf-8')) if args.v1_qa else build_questions(instances, args.seed)
    questions = update_size_gt(questions, instances, load_jsonl(args.frame_index))
    for question in questions:
        label, index = bin_answer(question['gt_value_m'], question['task'])
        if (question['answer_letter'], question['answer_label'], question['gt_bin']) != ('ABCD'[index], label, label):
            raise ValueError(f"Answer label does not match the reference interval for {question['id']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {len(questions)} questions: {args.output}')
    print(dict(Counter(question['task'] for question in questions)))


if __name__ == '__main__':
    main()
