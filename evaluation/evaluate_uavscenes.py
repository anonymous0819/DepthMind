#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavscenes_common import extract_letter, load_benchmark, read_predictions, write_atomic


def align_predictions(rows, predictions):
    selected = {row["id"]: row for row in rows}
    aligned = {}
    for prediction in predictions:
        sid = prediction["id"]
        if sid not in selected or sid in aligned:
            raise ValueError(f"Duplicate or out-of-benchmark prediction ID: {sid}")
        row = selected[sid]
        for key in ("question", "choices", "scene", "subset", "task"):
            if key in prediction and prediction[key] != row[key]:
                raise ValueError(f"Prediction {key} differs from authoritative QA: {sid}")
        response = prediction.get("response", prediction.get("turn2_answer"))
        if response is None:
            response = ""
        if not isinstance(response, str):
            raise ValueError(f"Response must be a string or null: {sid}")
        aligned[sid] = dict(response=response, branch=prediction.get("branch", "unknown"),
                            tool_status=(prediction.get("tool_result") or {}).get("status", "not_called"))
    return aligned


def evaluate(rows, predictions, answer_extractor):
    detailed = []
    for row in rows:
        sid = row["id"]
        present = sid in predictions
        prediction = predictions.get(sid, {})
        response = prediction.get("response", "")
        letter = extract_letter(response, answer_extractor)
        if not present:
            source = "missing_prediction"
        elif not response.strip():
            source = "empty_prediction"
        elif letter is None:
            source = "unparsed"
        else:
            source = "mcq_regex"
        correct = source == "mcq_regex" and letter == row["answer_letter"]
        detailed.append(dict(id=sid, task=row["task"], subset=row["subset"], scene=row["scene"],
                             question=row["question"], gt_letter=row["answer_letter"], gt_value_m=row["gt_value_m"],
                             gt_bin=row.get("gt_bin", row["choices"]["ABCD".index(row["answer_letter"])]),
                             response=response, pred_letter=letter, correct=correct, source=source,
                             branch=prediction.get("branch", "missing"), tool_status=prediction.get("tool_status", "not_called")))

    def stats(records):
        correct, total = sum(record["correct"] for record in records), len(records)
        return dict(correct=correct, total=total, accuracy=correct / total if total else 0.)

    def group_stats(keys):
        groups = {}
        for record in detailed:
            name = "/".join(str(record[key]) for key in keys)
            groups.setdefault(name, []).append(record)
        return {name: stats(records) for name, records in sorted(groups.items())}

    return dict(stats(detailed), answer_extractor=answer_extractor,
                per_task=group_stats(["task"]), per_subset=group_stats(["subset"]), per_scene=group_stats(["scene"]),
                per_task_subset=group_stats(["task", "subset"]), per_task_gt_bin=group_stats(["task", "gt_bin"]),
                sources=dict(Counter(record["source"] for record in detailed)),
                branches=dict(Counter(record["branch"] for record in detailed)),
                tool_statuses=dict(Counter(record["tool_status"] for record in detailed)), detailed=detailed)


def main():
    parser = argparse.ArgumentParser(description="QA-authoritative UAVScenes-Metric v2 MCQ evaluation; no LLM judge")
    parser.add_argument("--qa_file", default=str(Path(__file__).resolve().parents[1] / "benchmarks/uavscenes_metric_v2.json"))
    parser.add_argument("--responses", nargs="+", required=True, help="Response JSONL files with disjoint IDs")
    parser.add_argument("--answer_extractor", choices=("legacy-direct", "cot", "first-letter"), required=True,
                        help="legacy-direct for historical DRoPE; cot for step-by-step responses; first-letter is a separate parser")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--output")
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    rows = load_benchmark(args.qa_file, args.max_samples)
    predictions = align_predictions(rows, read_predictions(args.responses))
    print(f"QA: {len(rows)}; predictions: {len(predictions)}; missing: {len(rows) - len(predictions)}; "
          f"empty: {sum(not row['response'].strip() for row in predictions.values())}; parser: {args.answer_extractor}")
    if args.check_only:
        return
    if not args.output:
        parser.error("--output is required unless --check_only is used")
    output = Path(args.output)
    if output.resolve() in {Path(path).resolve() for path in [args.qa_file, *args.responses]}:
        parser.error("Output must not overwrite inputs")
    if output.exists():
        raise FileExistsError("Evaluation output exists; use a new output file")
    result = evaluate(rows, predictions, args.answer_extractor)
    result["config"] = dict(qa_file=str(Path(args.qa_file).resolve()),
                            responses=[str(Path(path).resolve()) for path in args.responses], max_samples=args.max_samples)
    write_atomic(output, result)
    print(f"Overall: {100. * result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
    for name, stats in result["per_task"].items():
        print(f"{name}: {100. * stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    print(f"Saved scores to {output}")


if __name__ == "__main__":
    main()
