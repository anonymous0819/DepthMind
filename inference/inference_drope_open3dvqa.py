#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcot.answer_prompts import extract_letter_cot
from drope_common import build_drope_messages
from tcot.qwen3vl_runtime import generate_text, load_generator_bundle


def iter_samples(data):
    for dataset, scenes in data.items():
        for scene, entries in scenes.items():
            for item in entries:
                yield dataset, scene, item


def original_qa(item):
    turns = item.get("conversation") or []
    question = item.get("query_question")
    if not question:
        question = next((turn["value"] for turn in turns if turn.get("from") == "human"), "")
        question = question.replace("<image>", "").strip()
    answer = next((turn["value"] for turn in turns if turn.get("from") == "gpt"), None)
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Missing question or ground truth: {item.get('id')}")
    return question, answer


def select_samples(data, mode, data_root, depth_root):
    selected, ids = [], set()
    for dataset, scene, item in iter_samples(data):
        sid = item.get("id")
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError(f"Invalid or duplicate sample ID: {sid}")
        ids.add(sid)
        if mode == "metric-mcq":
            if item.get("mcq_tier") not in {"T3a", "T3b"}:
                continue
            if item.get("mcq_numeric_option_mode") != "rank_random_log_ratio":
                raise ValueError(f"MCQ requires the T3ab rank-random protocol: {sid}")
            if item.get("mcq_answer") not in {"A", "B", "C", "D"} or not item.get("mcq_question"):
                raise ValueError(f"Invalid MCQ options: {sid}")
        for component in (dataset, scene):
            if Path(component).name != component or component in {".", ".."}:
                raise ValueError("Dataset and scene must be single directory names")
        relative = Path(item["image_info"]["image_path"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("image_info.image_path must be dataset/scene relative")
        if len(relative.parts) == 1:
            relative = Path("rgb") / relative
        question, answer = original_qa(item)
        sample = dict(item, query_question=question, original_answer=answer, dataset=dataset, scene=scene,
                      rgb_path=str(Path(data_root) / dataset / scene / relative),
                      depth_path=str(Path(depth_root) / dataset / scene / relative.with_suffix(".npz")))
        if not Path(sample["rgb_path"]).is_file() or not Path(sample["depth_path"]).is_file():
            raise FileNotFoundError(f"Missing RGB or DRoPE depth for {sid}")
        selected.append(sample)
    if not selected:
        raise ValueError(f"No samples for {mode}")
    return selected


def run_one(sample, args, bundle):
    from PIL import Image

    question = sample["mcq_question"] if args.mode == "metric-mcq" else sample["query_question"]
    with Image.open(sample["rgb_path"]) as handle:
        image = handle.convert("RGB")
    messages = build_drope_messages(image, question, bundle.experiment)
    response = generate_text(bundle, messages, sample["depth_path"], args.max_new_tokens)
    record = dict(id=sample["id"], image_name=sample.get("image_name") or Path(sample["rgb_path"]).name,
                  qa_info=sample.get("qa_info"), question=sample["query_question"],
                  answer=sample["original_answer"], response=response,
                  method="drope", experiment=bundle.experiment, mode=args.mode)
    if args.mode == "metric-mcq":
        record.update(mcq_question=sample["mcq_question"], mcq_answer=sample["mcq_answer"],
                      mcq_tier=sample["mcq_tier"], mcq_numeric_option_mode=sample["mcq_numeric_option_mode"],
                      pred_letter=extract_letter_cot(response))
    return record


def save_results(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description="Independent Qwen3-VL DRoPE inference without geometric tools")
    parser.add_argument("--mode", choices=("freeform", "metric-mcq"), default="freeform")
    parser.add_argument("--qa_file", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--estimated_depth_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_samples is not None and args.max_samples < 1:
        parser.error("Token budget and max_samples must be positive")
    output = Path(args.output)
    if output.resolve() == Path(args.qa_file).resolve():
        parser.error("Output must not overwrite the QA file")
    samples = select_samples(json.loads(Path(args.qa_file).read_text(encoding="utf-8")), args.mode,
                             args.data_dir, args.estimated_depth_root)
    if args.max_samples:
        samples = samples[:args.max_samples]
    records, finished = {}, set()
    selected = {sample["id"]: sample for sample in samples}
    if output.exists():
        if not args.resume:
            raise FileExistsError("Output exists; use --resume or a new output file")
        records = json.loads(output.read_text(encoding="utf-8"))
        for dataset, scene, row in iter_samples(records):
            sid = row["id"]
            if sid in finished or sid not in selected:
                raise ValueError(f"Duplicate saved ID or ID outside selected subset: {sid}")
            sample = selected[sid]
            if (row.get("method") != "drope" or row.get("mode") != args.mode
                    or (dataset, scene) != (sample["dataset"], sample["scene"])
                    or row.get("question") != sample["query_question"]
                    or row.get("answer") != sample["original_answer"]
                    or not isinstance(row.get("response"), str)):
                raise ValueError(f"Saved record does not match this DRoPE run: {sid}")
            if args.mode == "metric-mcq" and any(row.get(key) != sample[key] for key in ("mcq_question", "mcq_answer")):
                raise ValueError(f"Saved MCQ options do not match QA: {sid}")
            finished.add(sid)
    pending = [sample for sample in samples if sample["id"] not in finished]
    if pending:
        bundle = load_generator_bundle(args.model_path, args.adapter_path, args.max_new_tokens)
        try:
            for index, sample in enumerate(pending, 1):
                record = run_one(sample, args, bundle)
                records.setdefault(sample["dataset"], {}).setdefault(sample["scene"], []).append(record)
                print(f"[{len(finished) + index}/{len(samples)}] {sample['id']}", flush=True)
                if index % 5 == 0:
                    save_results(output, records)
        finally:
            save_results(output, records)
    else:
        save_results(output, records)
    print(f"Saved {sum(1 for _ in iter_samples(records))} records to {output}")


if __name__ == "__main__":
    main()
