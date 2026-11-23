#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from tcot import fusion_stages, metric_freeform, metric_mcq
from tcot.cache.cached_geometry import CachedGeometryBackend, METRIC_QUESTIONS
from tcot.o3dvqa_protocol import build_turn1_universal_tool_call_o3dvqa, build_universal_tool_first_system_prompt
from tcot.qwen3vl_runtime import generate_independent_answer, generate_text, load_generator_bundle, make_messages
from tcot.tool_schema import TOOL_O3D_RELATION_EVIDENCE, parse_tool_call


def iter_samples(data):
    for dataset, scenes in data.items():
        for scene, entries in scenes.items():
            for item in entries:
                yield dataset, scene, item


def select_samples(data, module, data_root, depth_root):
    selected = []
    ids = set()
    for dataset, scene, item in iter_samples(data):
        sid = item["id"]
        if sid in ids:
            raise ValueError(f"Duplicate sample ID: {sid}")
        ids.add(sid)
        metric = (item.get("qa_info") or {}).get("question_name") in METRIC_QUESTIONS
        if (module == "relation") == metric:
            continue
        if module == "metric-mcq":
            if item.get("mcq_tier") not in {"T3a", "T3b"} or item.get("mcq_numeric_option_mode") != "rank_random_log_ratio":
                raise ValueError(f"MCQ requires the T3ab rank-random protocol: {sid}")
            if item.get("mcq_answer") not in {"A", "B", "C", "D"} or not item.get("mcq_question"):
                raise ValueError(f"Invalid MCQ options: {sid}")
        relative = Path(item["image_info"]["image_path"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("image_info.image_path must be dataset/scene relative")
        if len(relative.parts) == 1:
            relative = Path("rgb") / relative
        question = item.get("query_question") or item["conversation"][0]["value"]
        sample = dict(item, query_question=question, dataset=dataset, scene=scene,
                      rgb_path=str(Path(data_root) / dataset / scene / relative),
                      depth_path=str(Path(depth_root) / dataset / scene / relative.with_suffix(".npz")))
        if not Path(sample["rgb_path"]).is_file() or not Path(sample["depth_path"]).is_file():
            raise FileNotFoundError(f"Missing RGB or DRoPE depth for {sid}")
        selected.append(sample)
    if not selected:
        raise ValueError(f"No samples for {module}")
    return selected


def run_one(sample, args, bundle, backend):
    qname = (sample.get("qa_info") or {}).get("question_name")
    question = sample["mcq_question"] if args.module == "metric-mcq" else sample["query_question"]
    with Image.open(sample["rgb_path"]) as handle:
        image = handle.convert("RGB")
    messages = make_messages(image, build_turn1_universal_tool_call_o3dvqa(question),
                             build_universal_tool_first_system_prompt())
    raw_call = generate_text(bundle, messages, sample["depth_path"], args.tool_tokens)
    call, status = parse_tool_call(raw_call)
    evidence, fusion_image, result = None, image, None
    if call is not None:
        if args.module == "relation" and call.name == TOOL_O3D_RELATION_EVIDENCE:
            evidence, fusion_image = backend.relation(sample, call, image)
            status = "ok" if evidence.get("status") != "failed" else evidence.get("reason", "failed")
        elif args.module != "relation" and call.name != TOOL_O3D_RELATION_EVIDENCE:
            result = backend.metric(sample, call)
            status = result.status
        else:
            status = "wrong_tool_family"
    if args.module == "relation":
        draft = generate_independent_answer(bundle, image, sample["query_question"], sample["depth_path"])
        output = fusion_stages.run_chain(bundle, image, sample, backend, evidence or {}, draft)
        output.update(direct_response=draft, relation_evidence=evidence,
                      three_stage=dict(direct_source="generated", max_new_tokens_direct=128,
                                       direct_context="original_drope_no_tool_history_or_evidence"))
    elif args.module == "metric-freeform":
        output = metric_freeform.run(bundle, image, question, sample["depth_path"], result, messages, raw_call)
    else:
        output = metric_mcq.run(bundle, image, question, qname, sample["depth_path"], result, messages, raw_call)
    output.update(id=sample["id"], image_name=sample["image_name"], qa_info=sample.get("qa_info"),
                  question=sample["query_question"], answer=sample["conversation"][1]["value"],
                  turn1_tool_pick=raw_call, tool_audit=dict(status=status, tool_call=asdict(call) if call else None))
    if result is not None:
        output["tool_result"] = result.to_record()
    if args.module == "metric-mcq":
        output.update(mcq_question=sample["mcq_question"], mcq_answer=sample["mcq_answer"],
                      mcq_tier=sample["mcq_tier"], original_question=sample["query_question"],
                      original_answer=sample["conversation"][1]["value"])
    return output


def save_results(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def build_run_config(args, samples, backend):
    import hashlib
    from dronepur_common import file_digest

    root = Path(__file__).resolve().parents[1]
    paths = {Path(sample[key]).resolve() for sample in samples for key in ("rgb_path", "depth_path")}
    for sample in samples:
        entry = backend.entries.get(sample["id"], {})
        if not isinstance(entry, dict):
            continue
        for key in ("depth_path", "intrinsics_path"):
            if entry.get(key):
                paths.add(backend.resolve(entry[key]).resolve())
        for obj in (entry.get("objects") or {}).values():
            if isinstance(obj, dict) and obj.get("mask_path"):
                paths.add(backend.resolve(obj["mask_path"]).resolve())
    assets = {str(path): file_digest(path) if path.is_file() else "missing" for path in sorted(paths)}
    model_files = {}
    for key in ("model_path", "adapter_path"):
        directory = Path(getattr(args, key)).resolve()
        if not directory.is_dir():
            raise ValueError(f"Expected a local model/adapter directory: {directory}")
        model_files[key] = {
            path.relative_to(directory).as_posix(): dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            for path in sorted(directory.rglob("*")) if path.is_file()
        }
    code = [Path(__file__).resolve(), root / "drope_common.py",
            root / "train_depth_rope_qwen3vl.py"] + sorted((root / "tcot").rglob("*.py"))
    return dict(protocol="qwen3_three_stage_v2_1_vertical_lateral_clock_binding_online_draft_v1", module=args.module,
                draft_tokens=128, draft_source="generated_original_drope_prompt",
                relation_cues_sha256=file_digest(backend.relation_path) if backend.relation_path else None,
                qa_sha256=file_digest(args.qa_file), perception_manifest_sha256=file_digest(backend.path),
                model_path=str(Path(args.model_path).resolve()), adapter_path=str(Path(args.adapter_path).resolve()),
                model_files=model_files, data_dir=str(Path(args.data_dir).resolve()),
                estimated_depth_root=str(Path(args.estimated_depth_root).resolve()),
                max_new_tokens=args.max_new_tokens, tool_tokens=args.tool_tokens, do_sample=False,
                assets_sha256=hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest(),
                selected_count=len(samples),
                selected_ids_sha256=hashlib.sha256(json.dumps([sample["id"] for sample in samples]).encode()).hexdigest(),
                code_sha256={str(path.relative_to(root)): file_digest(path) for path in code})


def validate_saved_record(dataset, scene, record, sample, config):
    expected = dict(image_name=sample["image_name"], qa_info=sample.get("qa_info"),
                    question=sample["query_question"], answer=sample["conversation"][1]["value"],
                    module=config["module"], run_config=config)
    if config["module"] == "metric-mcq":
        expected.update(mcq_question=sample["mcq_question"], mcq_answer=sample["mcq_answer"],
                        mcq_tier=sample["mcq_tier"])
    if ((dataset, scene) != (sample["dataset"], sample["scene"])
            or any(record.get(key) != value for key, value in expected.items())
            or not isinstance(record.get("response"), str)):
        raise ValueError(f"Saved record does not match this TCoT run; use a new output: {sample['id']}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL TCoT using precomputed masks, depth and intrinsics")
    parser.add_argument("--module", choices=("relation", "metric-freeform", "metric-mcq"), required=True)
    parser.add_argument("--qa_file", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--estimated_depth_root", required=True, help="DRoPE input depth, independent of tool depth")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--perception_manifest", required=True, help="ID-to-predicted-assets JSON for geometric tools")
    parser.add_argument("--relation_cues", help="Historical offline object-geometry cues, not response/draft records; required for relation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--tool_tokens", type=int, default=192)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--resume", action="store_true", help="Skip completed samples in this output")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.tool_tokens < 1 or args.max_samples is not None and args.max_samples < 1:
        parser.error("Token budgets and max_samples must be positive")
    if args.module == "relation" and not args.relation_cues:
        parser.error("--relation_cues is required for the historical relation fusion chain")
    samples = select_samples(json.loads(Path(args.qa_file).read_text(encoding="utf-8")), args.module,
                             args.data_dir, args.estimated_depth_root)
    if args.max_samples:
        samples = samples[:args.max_samples]
    output = Path(args.output)
    if output.resolve() in {Path(args.qa_file).resolve(), Path(args.perception_manifest).resolve()}:
        raise ValueError("Output must not overwrite QA or perception manifest")
    if output.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume or a new output file")
    backend = CachedGeometryBackend(args.perception_manifest, args.relation_cues)
    print(f"Selected {len(samples)} samples; verifying TCoT input identity before model loading", flush=True)
    config = build_run_config(args, samples, backend)
    records = {}
    finished = set()
    selected = {sample["id"]: sample for sample in samples}
    if output.exists():
        records = json.loads(output.read_text(encoding="utf-8"))
        for dataset, scene, row in iter_samples(records):
            sid = row["id"]
            if sid in finished:
                raise ValueError("Duplicate saved sample ID")
            if sid not in selected:
                raise ValueError("Output contains samples outside the selected subset")
            validate_saved_record(dataset, scene, row, selected[sid], config)
            finished.add(sid)
    pending = [sample for sample in samples if sample["id"] not in finished]
    if pending:
        bundle = load_generator_bundle(args.model_path, args.adapter_path, args.max_new_tokens)
        try:
            for index, sample in enumerate(pending, 1):
                record = run_one(sample, args, bundle, backend)
                record.update(module=args.module, run_config=config)
                records.setdefault(sample["dataset"], {}).setdefault(sample["scene"], []).append(record)
                finished.add(sample["id"])
                print(f"[{len(finished)}/{len(samples)}] {sample['id']}", flush=True)
                if index % 5 == 0:
                    save_results(output, records)
        finally:
            save_results(output, records)
    else:
        save_results(output, records)
    print(f"Saved {len(finished)} records to {output}")


if __name__ == "__main__":
    main()
