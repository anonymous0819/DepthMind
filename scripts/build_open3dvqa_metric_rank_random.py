#!/usr/bin/env python3
"""Build Open3DVQA T3a/T3b rank-random numeric MCQs.

Accepts the official full test QA or an existing numeric T3ab subset. Numerical
rank cycles within tier/question/unit buckets; option letters are shuffled.
Historical log-ratio sampling is retained; two-decimal collision handling
reserves the GT and adjusts distractors only. Very small GTs may not admit
all lower distractors at this precision, so numerical-rank balance is approximate.
All image/depth path fields in the exported QA contain filenames only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

LETTERS = ['A', 'B', 'C', 'D']
NUM_RE = re.compile(r'[-+]?\d*\.?\d+')
OPTION_RE = re.compile(r'^([A-D])\.\s*([-+]?\d+(?:\.\d+)?)\s+(meters|cubic meters)\s*$')
DISTANCE_TASKS = {
    'distance2agent', 'vertical_distance2agent', 'horizontal_distance2agent',
    'distance_data', 'vertical_distance_data', 'horizontal_distance_data',
}
SIZE_TASKS = {'width_data', 'height_data', 'volume_data'}


def filename(path):
    return str(path).replace('\\', '/').rsplit('/', 1)[-1]


def portable_paths(value):
    if isinstance(value, dict):
        return {key: filename(item) if key.endswith('_path') and isinstance(item, str) else portable_paths(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [portable_paths(item) for item in value]
    return value


def fmt_num(value):
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f'{value:.2f}'


def first_number(text):
    match = NUM_RE.search(text or '')
    return float(match.group(0)) if match else None


class RankRandomNumericBuilder:
    def __init__(self, seed, low_range, high_range):
        self.rng = random.Random(seed)
        self.low_range = low_range
        self.high_range = high_range
        self.bucket_counts = defaultdict(int)

    def _sample_log_uniform(self, low, high):
        return math.exp(self.rng.uniform(math.log(low), math.log(high)))

    def _next_low_count(self, bucket):
        index = self.bucket_counts[bucket]
        self.bucket_counts[bucket] += 1
        return index % 4

    def build_values(self, gt, bucket):
        if gt <= 0:
            scale = max(1.0, abs(gt))
            values = [gt] + [scale * self._sample_log_uniform(*self.high_range) for _ in range(3)]
            return self._dedupe(values, 0), 0
        low_count = self._next_low_count(bucket)
        lows = [gt * self._sample_log_uniform(*self.low_range) for _ in range(low_count)]
        highs = [gt * self._sample_log_uniform(*self.high_range) for _ in range(3 - low_count)]
        values = lows + [gt] + highs
        return self._dedupe(values, len(lows)), len(lows)

    def _dedupe(self, values, gt_index):
        gt = float(values[gt_index])
        if not math.isfinite(gt) or gt < 0:
            raise ValueError('GT must be finite and nonnegative')
        seen = {round(gt, 2)}
        output = []
        for index, value in enumerate(values):
            if index == gt_index:
                output.append(gt)
                continue
            value = max(0.0, float(value))
            key = round(value, 2)
            tries = 0
            while key in seen and tries < 50:
                jitter = self.rng.uniform(0.03, 0.15) * max(1.0, abs(gt))
                if value < gt:
                    value = max(0.0, value - jitter)
                else:
                    value += jitter
                key = round(value, 2)
                tries += 1
            if key in seen:
                step = 0.01 * max(1.0, abs(gt))
                value = gt
                while round(value, 2) in seen:
                    value += step
            seen.add(round(value, 2))
            output.append(value)
        return output

    def build_lettered(self, gt, bucket, unit):
        values, gt_index = self.build_values(gt, bucket)
        indexed = list(enumerate(values))
        self.rng.shuffle(indexed)
        options = []
        gt_letter = ''
        for letter, (original_index, value) in zip(LETTERS, indexed):
            options.append((letter, value, f'{fmt_num(value)} {unit}'))
            if original_index == gt_index:
                gt_letter = letter
        if not gt_letter:
            raise RuntimeError('Failed to assign GT letter')
        sorted_desc = sorted(options, key=lambda item: item[1], reverse=True)
        rank = 1 + [item[0] for item in sorted_desc].index(gt_letter)
        return options, gt_letter, rank


def strip_options(question):
    return '\n'.join(line for line in (question or '').splitlines()
                     if not line.startswith('Please directly output') and not OPTION_RE.match(line.strip())).rstrip()


def build_benchmark(data, seed=20261101, low_range=(0.25, 0.70), high_range=(1.40, 4.00)):
    if not (0 < low_range[0] <= low_range[1] < 1 < high_range[0] <= high_range[1]):
        raise ValueError('Require 0 < low ratios < 1 < high ratios')
    builder = RankRandomNumericBuilder(seed, low_range, high_range)
    output = {}
    ids = set()
    for dataset, scenes in data.items():
        output[dataset] = {}
        for scene, items in scenes.items():
            selected = []
            for item in items:
                qname = (item.get('qa_info') or {}).get('question_name')
                if qname not in DISTANCE_TASKS | SIZE_TASKS:
                    continue
                if item['id'] in ids:
                    raise ValueError(f"Duplicate ID: {item['id']}")
                ids.add(item['id'])
                tier = 'T3a' if qname in DISTANCE_TASKS else 'T3b'
                if item.get('mcq_tier', tier) != tier:
                    raise ValueError(f"Wrong numeric tier for {item['id']}")
                conversation = item.get('conversation', [])
                text = conversation[1].get('value', '') if len(conversation) > 1 else ''
                gt = first_number(text)
                if gt is None or not math.isfinite(gt) or gt < 0:
                    raise ValueError(f"Invalid metric GT for {item['id']}: {text}")
                unit = 'cubic meters' if qname == 'volume_data' else 'meters'
                options, letter, rank = builder.build_lettered(gt, f'o3d:{tier}:{qname}:{unit}', unit)
                question = strip_options(item.get('mcq_question') or item.get('query_question') or '')
                options_text = '\n'.join(f'{label}. {value}' for label, _, value in options)
                suffix = '\nPlease directly output the single letter of the correct option (A, B, C or D). Do not output anything else.'
                result = portable_paths(deepcopy(item))
                image_name = filename(result['image_info']['image_path'])
                result.update(image_name=image_name, mcq_tier=tier, mcq_question=f'{question}\n{options_text}{suffix}',
                              mcq_answer=letter, mcq_numeric_option_mode='rank_random_log_ratio', mcq_numeric_gt_rank_desc=rank)
                selected.append(result)
            if selected:
                output[dataset][scene] = selected
        if not output[dataset]:
            del output[dataset]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Official test_qa.json or numeric T3ab QA JSON.')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'benchmarks/open3dvqa_metric_rank_random.json')
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--low-range', type=float, nargs=2, default=(0.25, 0.70))
    parser.add_argument('--high-range', type=float, nargs=2, default=(1.40, 4.00))
    args = parser.parse_args()
    data = build_benchmark(json.loads(args.input.read_text(encoding='utf-8')), args.seed, args.low_range, args.high_range)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    items = [item for scenes in data.values() for entries in scenes.values() for item in entries]
    print(f'Saved {len(items)} metric questions: {args.output}')
    print('Tiers:', dict(Counter(item['mcq_tier'] for item in items)))
    print('GT ranks (descending):', dict(sorted(Counter(item['mcq_numeric_gt_rank_desc'] for item in items).items())))


if __name__ == '__main__':
    main()
