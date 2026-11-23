#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcot.answer_prompts import extract_letter_cot

JUDGE_SYSTEM_PROMPT = "You are a helpful assistant designed to evaluate QA quality."
CATEGORY_ORDER = ("Allo.Size", "Allo.Distance", "Ego.Direction", "Ego.Distance",
                  "Trans.Direction", "Trans.Distance", "Obj.Size")
METRIC_QUESTIONS = {"width_data", "height_data", "volume_data", "distance2agent",
                    "horizontal_distance2agent", "vertical_distance2agent", "distance_data",
                    "horizontal_distance_data", "vertical_distance_data"}


def build_default_judge_prompt(question, answer, response):
    return f"""
You should help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 0 and 1.
1 means that the response perfectly matches the answer.
0 means that the response is completely different from the answer.
Only output 0 or 1. Do not output any explanation.

Your Turn:
Question: {question}
Answer: {answer}
Response: {response}
""".strip()


def extract_binary_score(text):
    normalized = str(text).strip()
    if normalized in {"0", "1"}:
        return int(normalized)
    match = re.search(r"\b[01]\b", normalized)
    if match:
        return int(match.group(0))
    leading = re.match(r"\s*([A-Za-z]+)", normalized)
    if leading:
        head = leading.group(1).lower()
        if head in {"yes", "y", "true", "correct", "right", "match", "matched", "matches", "equivalent", "same"}:
            return 1
        if head in {"no", "n", "false", "incorrect", "wrong", "mismatch", "different", "differs", "nope"}:
            return 0
    return None


def extract_mcq_letter(response, answer_extractor="cot"):
    if answer_extractor == "cot":
        return extract_letter_cot(response)
    if answer_extractor != "first-letter":
        raise ValueError(f"Unknown answer extractor: {answer_extractor}")
    text = (response or "").strip()
    match = re.match(r"^\s*([ABCD])\b", text, re.IGNORECASE)
    if match is None:
        match = re.search(r"\b([ABCD])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def iter_samples(data):
    for dataset, scenes in data.items():
        for scene, entries in scenes.items():
            for item in entries:
                yield dataset, scene, item


def load_rows(paths):
    rows = {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for dataset, scene, item in iter_samples(data):
            sid = item.get("id")
            if not isinstance(sid, str) or not sid or sid in rows:
                raise ValueError(f"Invalid or duplicate ID in {path}: {sid}")
            rows[sid] = (dataset, scene, item)
    return rows


def original_qa(item):
    turns = item.get("conversation") or []
    question = item.get("query_question")
    if not question:
        question = next((turn["value"] for turn in turns if turn.get("from") == "human"), "")
        question = question.replace("<image>", "").strip()
    answer = next((turn["value"] for turn in turns if turn.get("from") == "gpt"), None)
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Missing question or ground truth in QA: {item.get('id')}")
    return question, answer


def classify_item(item, question):
    name = (item.get("qa_info") or {}).get("question_name") or ""
    base = re.sub(r"_(predicate|choice|tfqa|multichoice)$", "", name)
    if base in {"wide", "thin", "tall", "short", "big", "small"}:
        return "Allo.Size"
    if name in {"distance_data", "horizontal_distance_data", "vertical_distance_data"}:
        return "Allo.Distance"
    if base in {"left", "right", "above", "below", "front", "behind"} or name in {
        "direction2agent", "angle2agent", "left_relationship2agent", "right_relationship2agent",
        "above_relationship2agent", "below_relationship2agent",
    }:
        return "Ego.Direction"
    if name == "distance2agent":
        return "Ego.Distance"
    if name in {"direction_data", "angle_data"}:
        return "Trans.Direction"
    if name in {"horizontal_distance2agent", "vertical_distance2agent"}:
        return "Trans.Distance"
    if name in {"width_data", "height_data", "volume_data"}:
        return "Obj.Size"
    text = question.lower()
    if any(word in text for word in ("wider", "thinner", "taller", "shorter", "bigger", "smaller",
                                    "more height", "less height", "more width", "less width", "more volume", "less volume")):
        return "Allo.Size"
    if any(word in text for word in ("left", "right", "above", "below", "under", "over", "in front", "before",
                                    "behind", "closer", "further", "distant", "back of")):
        return "Ego.Direction"
    return "Other.Other"


def prepare_samples(qa_rows, predictions, mode, mcq_rows=None):
    if not qa_rows:
        raise ValueError("QA is empty")
    extras = predictions.keys() - qa_rows.keys()
    if extras:
        raise ValueError(f"Predictions contain {len(extras)} IDs outside QA: {sorted(extras)[:5]}")
    mcq_rows = mcq_rows or {}
    if mode != "hybrid" and mcq_rows:
        raise ValueError("MCQ QA overlay is only supported in hybrid mode")
    for sid, (dataset, scene, item) in mcq_rows.items():
        if sid not in qa_rows or item.get("mcq_tier") not in {"T3a", "T3b"}:
            raise ValueError(f"MCQ overlay must contain only QA-matched T3a/T3b IDs: {sid}")
        base_dataset, base_scene, base_item = qa_rows[sid]
        question, answer = original_qa(item)
        base_question, base_answer = original_qa(base_item)
        if ((dataset, scene) != (base_dataset, base_scene)
                or (question.strip(), answer.strip()) != (base_question.strip(), base_answer.strip())
                or (item.get("qa_info") or {}).get("question_name") != (base_item.get("qa_info") or {}).get("question_name")):
            raise ValueError(f"MCQ overlay does not match original QA: {sid}")
    samples = []
    for sid, (dataset, scene, item) in qa_rows.items():
        question, answer = original_qa(item)
        mcq_item = mcq_rows[sid][2] if sid in mcq_rows else item
        tier = mcq_item.get("mcq_tier") or "unknown"
        qname = (item.get("qa_info") or {}).get("question_name")
        if mode == "hybrid" and qname in METRIC_QUESTIONS and tier == "unknown":
            raise ValueError(f"Hybrid requires T3a/T3b metadata/options for metric ID {sid}; provide --mcq_qa_file")
        scoring = "mcq" if mode == "hybrid" and tier in {"T3a", "T3b"} else "freeform"
        category = classify_item(item, question)
        if scoring == "mcq":
            question, answer = mcq_item.get("mcq_question"), mcq_item.get("mcq_answer")
            if not isinstance(question, str) or not question.strip() or answer not in {"A", "B", "C", "D"}:
                raise ValueError(f"Missing MCQ question or ground truth: {sid}")
        present = sid in predictions
        response = ""
        if present:
            pred_dataset, pred_scene, prediction = predictions[sid]
            if (pred_dataset, pred_scene) != (dataset, scene):
                raise ValueError(f"Prediction dataset/scene differs from QA: {sid}")
            response = prediction.get("response")
            if response is None:
                response = ""
            if not isinstance(response, str):
                raise ValueError(f"Prediction response must be a string or null: {sid}")
            if scoring == "mcq" and prediction.get("mcq_question") is not None:
                if prediction["mcq_question"].strip() != question.strip():
                    raise ValueError(f"Prediction MCQ options differ from authoritative QA: {sid}")
        samples.append(dict(id=sid, dataset=dataset, scene=scene, category=category, tier=tier,
                            scoring=scoring, question=question, answer=answer, response=response,
                            prediction_present=present))
    return samples


class LocalJudge:
    def __init__(self, model_path, max_new_tokens=8):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Judge must be an existing local Qwen3-VL model directory: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, question, answer, response):
        import torch

        messages = [
            {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": build_default_judge_prompt(question, answer, response)}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()


def score_sample(sample, judge=None, answer_extractor="cot", max_retries=3):
    record = dict(sample, score=0, judge_outputs=[])
    if not sample["prediction_present"]:
        record["source"] = "missing_prediction"
    elif not sample["response"].strip():
        record["source"] = "empty_prediction"
    elif sample["scoring"] == "mcq":
        letter = extract_mcq_letter(sample["response"], answer_extractor)
        record.update(pred_letter=letter, score=int(letter == sample["answer"]),
                      source="mcq_regex" if letter else "mcq_unparsed")
    else:
        if judge is None:
            raise ValueError("A local judge is required for nonempty free-form predictions")
        record["source"] = "judge_unparsed"
        for _ in range(max_retries):
            text = judge(sample["question"], sample["answer"], sample["response"])
            record["judge_outputs"].append(text)
            score = extract_binary_score(text)
            if score is not None:
                record.update(score=score, source="llm_judge")
                break
    return record


def summarize(scores, expected_total):
    def stats(rows):
        correct, total = sum(row["score"] for row in rows), len(rows)
        return dict(correct=correct, total=total, accuracy=correct / total if total else 0.)

    summary = dict(stats(scores), expected_total=expected_total, complete=len(scores) == expected_total)
    summary["categories"] = {name: stats([row for row in scores if row["category"] == name])
                             for name in (*CATEGORY_ORDER, "Other.Other")}
    summary["datasets"] = {name: stats([row for row in scores if row["dataset"] == name])
                           for name in sorted({row["dataset"] for row in scores})}
    summary["tiers"] = {name: stats([row for row in scores if row["tier"] == name])
                        for name in sorted({row["tier"] for row in scores})}
    summary["sources"] = dict(Counter(row["source"] for row in scores))
    return summary


def save_evaluation(path, config, scores, expected_total, log_path=None):
    summary = summarize(scores, expected_total)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(config=config, summary=summary, scores=scores),
                                    ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
    if log_path:
        correct, lines = 0, []
        for index, row in enumerate(scores, 1):
            correct += row["score"]
            lines.append(f"Step {index}: Accuracy = {correct / index:.4f} ({correct}/{index}) ID={row['id']}")
        lines.append(f"Final: Accuracy = {summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = log_path.with_name(log_path.name + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, log_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="ID-aligned Open3DVQA free-form/hybrid evaluation with a fixed local judge")
    parser.add_argument("--mode", choices=("freeform", "hybrid"), required=True)
    parser.add_argument("--qa_file", required=True, help="Authoritative QA defining the evaluation population and GT")
    parser.add_argument("--responses", nargs="+", required=True, help="Nested response JSON files with disjoint IDs")
    parser.add_argument("--mcq_qa_file", help="Matching T3a/T3b QA overlay for hybrid; unnecessary if qa_file already contains options")
    parser.add_argument("--judge_model_path", help="Local base Qwen3-VL model directory; no DRoPE adapter")
    parser.add_argument("--answer_extractor", choices=("cot", "first-letter"), default="cot")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--output", help="JSON containing config, summary and one score record per QA ID")
    parser.add_argument("--log_file", help="Optional cumulative Step log in QA order")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check_only", action="store_true", help="Validate inputs/coverage/routing without loading a judge or writing scores")
    args = parser.parse_args()
    if min(args.max_new_tokens, args.max_retries, args.save_every) < 1:
        parser.error("Token budget, retries and save interval must be positive")
    if args.mcq_qa_file and args.mode != "hybrid":
        parser.error("--mcq_qa_file is only supported with --mode hybrid")
    qa_rows = load_rows([args.qa_file])
    predictions = load_rows(args.responses)
    mcq_rows = load_rows([args.mcq_qa_file]) if args.mcq_qa_file else None
    samples = prepare_samples(qa_rows, predictions, args.mode, mcq_rows)
    print(f"QA: {len(samples)}; predictions: {len(predictions)}; "
          f"missing: {sum(not row['prediction_present'] for row in samples)}; "
          f"empty: {sum(row['prediction_present'] and not row['response'].strip() for row in samples)}; "
          f"routing: {dict(Counter(row['scoring'] for row in samples))}", flush=True)
    if args.check_only:
        return
    if not args.output:
        parser.error("--output is required unless --check_only is used")
    output = Path(args.output)
    log_path = Path(args.log_file) if args.log_file else None
    inputs = {Path(path).resolve() for path in [args.qa_file, *args.responses, *([args.mcq_qa_file] if args.mcq_qa_file else [])]}
    if output.resolve() in inputs or log_path and (log_path.resolve() in inputs or log_path.resolve() == output.resolve()):
        parser.error("Evaluation output/log must not overwrite inputs or each other")
    if log_path and log_path.exists() and not args.resume:
        raise FileExistsError("Log exists; use --resume or a new log file")
    config = dict(mode=args.mode, qa_file=str(Path(args.qa_file).resolve()),
                  responses=[str(Path(path).resolve()) for path in args.responses],
                  mcq_qa_file=str(Path(args.mcq_qa_file).resolve()) if args.mcq_qa_file else None,
                  judge_model_path=str(Path(args.judge_model_path).resolve()) if args.judge_model_path else None,
                  judge_system_prompt=JUDGE_SYSTEM_PROMPT, judge_prompt_style="default",
                  answer_extractor=args.answer_extractor, max_new_tokens=args.max_new_tokens, max_retries=args.max_retries)
    completed = {}
    selected = {sample["id"]: sample for sample in samples}
    if output.exists():
        if not args.resume:
            raise FileExistsError("Output exists; use --resume or a new output file")
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved.get("config") != config:
            raise ValueError("Saved evaluation configuration differs from this run")
        for row in saved["scores"]:
            sid = row["id"]
            if sid in completed or sid not in selected or row.get("score") not in {0, 1}:
                raise ValueError(f"Invalid saved score or duplicate/extra ID: {sid}")
            if any(row.get(key) != value for key, value in selected[sid].items()):
                raise ValueError(f"Saved score inputs differ from current QA/response: {sid}")
            completed[sid] = row
    pending = [sample for sample in samples if sample["id"] not in completed]
    needs_judge = any(row["scoring"] == "freeform" and row["response"].strip() for row in pending)
    if needs_judge and not args.judge_model_path:
        parser.error("--judge_model_path is required for nonempty free-form predictions")
    judge = LocalJudge(args.judge_model_path, args.max_new_tokens) if needs_judge else None
    try:
        for index, sample in enumerate(pending, 1):
            record = score_sample(sample, judge, args.answer_extractor, args.max_retries)
            completed[sample["id"]] = record
            print(f"[{len(completed)}/{len(samples)}] {sample['id']}: {record['score']} ({record['source']})", flush=True)
            if index % args.save_every == 0:
                scores = [completed[row["id"]] for row in samples if row["id"] in completed]
                save_evaluation(output, config, scores, len(samples), log_path)
    finally:
        scores = [completed[row["id"]] for row in samples if row["id"] in completed]
        summary = save_evaluation(output, config, scores, len(samples), log_path)
    print(f"Overall: {100. * summary['accuracy']:.2f}% ({summary['correct']}/{summary['total']})")
    for name in CATEGORY_ORDER:
        stats = summary["categories"][name]
        print(f"{name}: {100. * stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    print(f"Saved scores to {output}")


if __name__ == "__main__":
    main()
