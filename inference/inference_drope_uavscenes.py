#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcot.qwen3vl_runtime import generate_text, make_messages
from uavscenes_common import MCQ_SYSTEM_PROMPT, extract_letter, inference_parser, run_inference


def build_messages(image, question, experiment):
    from train_depth_rope_qwen3vl import build_experiment_prompt

    return make_messages(image, build_experiment_prompt(question, experiment), MCQ_SYSTEM_PROMPT)


def run_one(bundle, image, row, depth_path, args, backend=None):
    messages = build_messages(image, row["question"], bundle.experiment)
    response = generate_text(bundle, messages, depth_path, args.max_new_tokens)
    return dict(response=response, pred_letter=extract_letter(response, "legacy-direct"), branch="direct")


def main():
    args = inference_parser("Independent Qwen3-VL DRoPE inference for UAVScenes-Metric v2").parse_args()
    run_inference(args, "drope", run_one)


if __name__ == "__main__":
    main()
