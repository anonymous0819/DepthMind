import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


SYSTEM_PROMPT = """You are a precise visual target extraction assistant for UAV visual question answering.
Your task is to extract only the visual objects, regions, or object categories that are needed to answer the question.
Do not answer the question. Do not infer which option is correct. Return valid JSON only.
"""

EXTRACTION_INSTRUCTION = """Extract question-grounded visual targets from the UAV scene question.

Return exactly one JSON object with this schema:
{
  "targets": {
    "subject": "one primary target phrase from the question, or empty string",
    "reference": "one reference target phrase from the question, or empty string"
  },
  "relations": [
    {
      "type": "left_of|right_of|above|below|above_comparison|below_comparison|nearer_than|farther_than|distance_to_camera|count|existence|attribute|other",
      "subject": "same phrase as targets.subject when applicable",
      "object": "same phrase as targets.reference when applicable"
    }
  ]
}

Rules:
- Extract only the target phrase(s) needed for grounding; do not answer the question.
- Do not generate prompt variants, synonyms, shortened forms, or extra SAM prompts.
- Keep at most one subject and at most one reference.
- For relation questions like "Is A to the left of B?", set subject=A and reference=B.
- For comparison questions like "Which is above, A or B?", set subject=A and reference=B without deciding the answer.
- For single-target counting, existence, distance, or attribute questions, set subject to the queried object/category and reference to an empty string.
- Preserve the original descriptive target phrase from the question as much as possible.
- Do not include non-visual words such as "which", "does", "appear", "from the viewer's perspective", or answer words.
- If no concrete visual target exists, return empty subject, empty reference, and empty relations.
- Return JSON only. No markdown.

Example:
Question: Which is above, the cluster of rectangular buildings or the dense green vegetation covering ground?
Output:
{
  "targets": {
    "subject": "the cluster of rectangular buildings",
    "reference": "dense green vegetation"
  },
  "relations": [
    {
      "type": "above_comparison",
      "subject": "the cluster of rectangular buildings",
      "object": "dense green vegetation"
    }
  ]
}
"""


DEFAULT_RESULT_PATH = str(Path(__file__).resolve().parents[2] / "target_extraction_result/qwen3_vl_8b_test_targets.json")


def count_qa(obj):
    count = 0
    if isinstance(obj, dict):
        if "id" in obj and "query_question" in obj:
            count += 1
        for value in obj.values():
            count += count_qa(value)
    elif isinstance(obj, list):
        for item in obj:
            count += count_qa(item)
    return count


def iter_qa_items(data):
    if isinstance(data, dict):
        for dataset, scenes in data.items():
            if isinstance(scenes, dict):
                for scene, entries in scenes.items():
                    if isinstance(entries, list):
                        for index, item in enumerate(entries):
                            if isinstance(item, dict) and "id" in item:
                                yield dataset, scene, index, item
                    else:
                        for sub_dataset, sub_scene, index, item in iter_qa_items(entries):
                            yield dataset, f"{scene}/{sub_scene}", index, item
            elif isinstance(scenes, list):
                for index, item in enumerate(scenes):
                    if isinstance(item, dict) and "id" in item:
                        yield dataset, "", index, item
    elif isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, dict) and "id" in item:
                yield "", "", index, item


def load_existing_results(result_path):
    if not os.path.exists(result_path):
        return defaultdict(lambda: defaultdict(list)), set()

    with open(result_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = defaultdict(lambda: defaultdict(list))
    finished_ids = set()
    for dataset, scenes in data.items():
        if not isinstance(scenes, dict):
            continue
        for scene, items in scenes.items():
            if not isinstance(items, list):
                continue
            results[dataset][scene].extend(items)
            for item in items:
                item_id = item.get("id")
                if item_id:
                    finished_ids.add(item_id)
    return results, finished_ids


def save_results(results, result_path):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    serializable = {dataset: dict(scenes) for dataset, scenes in results.items()}
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(serializable, file, ensure_ascii=False, indent=4)


def build_messages(question):
    user_text = f"{EXTRACTION_INSTRUCTION}\n\nQuestion: {question}\nOutput:"
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]


def extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as exc:
            return None, str(exc)

    return None, "No JSON object found in model output."


def normalize_extraction(parsed):
    if not isinstance(parsed, dict):
        parsed = {}

    raw_targets = parsed.get("targets", {})
    subject = ""
    reference = ""

    if isinstance(raw_targets, dict):
        subject = str(raw_targets.get("subject", "")).strip()
        reference = str(raw_targets.get("reference", "")).strip()
    elif isinstance(raw_targets, list):
        for target in raw_targets:
            if not isinstance(target, dict):
                continue
            phrase = str(target.get("phrase", "")).strip()
            if not phrase:
                continue
            role = str(target.get("role", "")).strip().lower()
            if not subject and role in {"subject", "candidate", "target", "category"}:
                subject = phrase
            elif not reference and role in {"reference", "object", "candidate", "background"}:
                reference = phrase
            elif not subject:
                subject = phrase
            elif not reference:
                reference = phrase

    relations = parsed.get("relations", [])
    if not isinstance(relations, list):
        relations = []

    normalized_relations = []
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        relation_subject = str(relation.get("subject", "")).strip()
        relation_object = str(relation.get("object", "")).strip()
        if not subject and relation_subject:
            subject = relation_subject
        if not reference and relation_object:
            reference = relation_object
        normalized_relations.append(
            {
                "type": str(relation.get("type", "other")).strip() or "other",
                "subject": relation_subject,
                "object": relation_object,
            }
        )

    for relation in normalized_relations:
        if not relation["subject"]:
            relation["subject"] = subject
        if not relation["object"]:
            relation["object"] = reference

    if not normalized_relations and (subject or reference):
        normalized_relations.append(
            {
                "type": "other",
                "subject": subject,
                "object": reference,
            }
        )

    return {
        "targets": {
            "subject": subject,
            "reference": reference,
        },
        "relations": normalized_relations,
    }


def generate_extraction(model, processor, question, max_new_tokens):
    messages = build_messages(question)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Extract question-grounded target objects from O3DVQA_v2/test_qa.json using Qwen3-VL.")
    parser.add_argument(
        "--qa_file",
        type=str,
        default="/root/src/UAV_reasoning/O3DVQA_v2/test_qa.json",
        help="Path to O3DVQA_v2/test_qa.json.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/src/pretrained_model/Qwen3-VL-8B-Instruct",
        help="Path to local Qwen3-VL-8B-Instruct model.",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default=DEFAULT_RESULT_PATH,
        help="Output JSON path for extracted targets.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum generated tokens per question.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional smoke-test limit. Defaults to all samples.")
    parser.add_argument("--save_every", type=int, default=20, help="Save intermediate results every N processed samples.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing result JSON.")
    args = parser.parse_args()

    with open(args.qa_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if args.resume:
        results, finished_ids = load_existing_results(args.result_path)
    else:
        results = defaultdict(lambda: defaultdict(list))
        finished_ids = set()

    total = count_qa(data)
    if args.max_samples is not None and args.max_samples > 0:
        total = min(total, args.max_samples)

    processed_count = 0
    progress_bar = tqdm(total=total, desc="Extracting targets")

    for dataset, scene, index, item in iter_qa_items(data):
        item_id = item.get("id")
        if item_id in finished_ids:
            progress_bar.update(1)
            continue

        question = item.get("query_question") or ""
        raw_output = generate_extraction(
            model=model,
            processor=processor,
            question=question,
            max_new_tokens=args.max_new_tokens,
        )
        parsed, parse_error = extract_json_object(raw_output)
        extraction = normalize_extraction(parsed)

        response = {
            "id": item_id,
            "image_name": item.get("image_name"),
            "image_info": item.get("image_info", {}),
            "qa_info": item.get("qa_info", {}),
            "question": question,
            "target_extraction": extraction,
            "raw_response": raw_output,
            "parse_error": parse_error,
        }
        results[dataset][scene].append(response)
        if item_id:
            finished_ids.add(item_id)

        processed_count += 1
        progress_bar.update(1)

        if processed_count % args.save_every == 0:
            save_results(results, args.result_path)

        if args.max_samples is not None and args.max_samples > 0 and processed_count >= args.max_samples:
            break

    progress_bar.close()
    save_results(results, args.result_path)
    print(f"Saved target extraction results to {args.result_path}")


if __name__ == "__main__":
    main()
