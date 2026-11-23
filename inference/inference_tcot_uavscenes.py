#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcot import metric_mcq
from tcot.qwen3vl_runtime import generate_text, make_messages
from tcot.cache.uavscenes import (
    TASK_TO_QNAME, TOOL_FIRST_SYSTEM_PROMPT, UavscenesToolBackend, build_tool_selection_prompt, parse_tool_call,
)
from uavscenes_common import inference_parser, run_inference


def make_backend(args):
    return UavscenesToolBackend(args.bbox_file, args.tool_depth_root, args.intrinsics_root, args.data_dir)


def run_one(bundle, image, row, depth_path, args, backend):
    messages = make_messages(image, build_tool_selection_prompt(row["question"]), TOOL_FIRST_SYSTEM_PROMPT)
    raw_call = generate_text(bundle, messages, depth_path, args.tool_tokens)
    call, parse_status = parse_tool_call(raw_call)
    result = backend.call(row, call) if call is not None else None
    output = metric_mcq.run(bundle, image, row["question"], TASK_TO_QNAME[row["task"]], depth_path,
                            result, messages, raw_call)
    output.update(turn1_tool_pick=raw_call, turn2_answer=output["response"],
                  branch="tool_ok" if result is not None and result.status == "ok" else "cot_fallback",
                  tool_call=dict(parse_status=parse_status, **asdict(call)) if call else dict(parse_status=parse_status),
                  tool_result=result.to_record() if result is not None else None)
    return output


def main():
    parser = inference_parser("Qwen3-VL UAVScenes-Metric tool-assisted reasoning (no margin gate)", tcot=True)
    args = parser.parse_args()
    run_inference(args, "drope_tcot", run_one, make_backend)


if __name__ == "__main__":
    main()
