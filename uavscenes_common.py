from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path

from tcot.answer_prompts import extract_letter_cot

BINS = {
    "abs_distance_ego": ([0, 50, 80, 110, math.inf], ["<50 m", "50-80 m", "80-110 m", ">110 m"]),
    "abs_size_horiz": ([0, 5, 10, 20, math.inf], ["<5 m", "5-10 m", "10-20 m", ">20 m"]),
    "abs_distance_obj": ([0, 20, 50, 100, math.inf], ["<20 m", "20-50 m", "50-100 m", ">100 m"]),
}
MCQ_SYSTEM_PROMPT = (
    "You are an assistant that answers spatial-reasoning multiple-choice "
    "questions about urban / UAV imagery. Every question below is a "
    "multiple-choice question with 2 or 4 options (A, B, C, D). "
    "Read the image and the question carefully, then output **only** the "
    "single letter (A, B, C or D) of the correct option. "
    "Do not output any explanation, punctuation or extra text."
)


def read_predictions(paths):
    rows, ids = [], set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object in {path}:{number}")
                sid = row.get("id")
                if not isinstance(sid, str) or not sid or sid in ids:
                    raise ValueError(f"Invalid or duplicate ID in {path}:{number}: {sid}")
                ids.add(sid)
                rows.append(row)
    return rows


def load_benchmark(path, max_samples=None):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("UAVScenes benchmark must be a nonempty JSON list")
    ids = set()
    for row in rows:
        sid = row.get("id")
        task = row.get("task")
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError(f"Invalid or duplicate benchmark ID: {sid}")
        ids.add(sid)
        if task not in BINS:
            raise ValueError(f"Unsupported UAVScenes task: {task}")
        for key in ("scene", "subset", "image_path", "question"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing {key}: {sid}")
        scene = row["scene"]
        if Path(scene).name != scene or scene in {".", ".."}:
            raise ValueError(f"Scene must be a single directory name: {sid}")
        edges, labels = BINS[task]
        if row.get("choices") != labels or row.get("answer_letter") not in {"A", "B", "C", "D"}:
            raise ValueError(f"Requires the frozen UAVScenes fixed-interval options: {sid}")
        option_lines = re.findall(r"^\(([ABCD])\)\s*(.*?)\s*$", row["question"], re.MULTILINE)
        if option_lines != list(zip("ABCD", labels)):
            raise ValueError(f"Question options do not match choices: {sid}")
        value = row.get("gt_value_m")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"Missing finite authoritative GT: {sid}")
        indices = [i for i in range(4) if edges[i] < value < edges[i + 1]]
        if len(indices) != 1 or row["answer_letter"] != "ABCD"[indices[0]]:
            raise ValueError(f"GT is on a strict interval boundary or has an incorrect letter: {sid}")
        label = labels[indices[0]]
        if any(key in row and row[key] != label for key in ("answer_label", "gt_bin")):
            raise ValueError(f"GT label metadata does not match the interval: {sid}")
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        rows = rows[:max_samples]
    return rows


def extract_letter(response, protocol):
    if protocol == "cot":
        return extract_letter_cot(response)
    if protocol == "legacy-direct":
        match = re.search(r"[A-D]", (response or "").upper())
    elif protocol == "first-letter":
        match = re.search(r"\b([A-D])\b", response or "", re.IGNORECASE)
    else:
        raise ValueError(f"Unknown letter parser: {protocol}")
    return match.group(0).upper() if match else None


def resolve_assets(row, data_dir, depth_root):
    data_root = Path(data_dir).resolve()
    text = row["image_path"].replace("\\", "/")
    image_ref = Path(text)
    if ".." in image_ref.parts:
        raise ValueError(f"Unsafe image reference: {row['id']}")
    expected_relative = Path(f"interval5_{row['scene']}") / "interval5_CAM" / image_ref.name
    if image_ref.is_absolute():
        try:
            relative = image_ref.resolve().relative_to(data_root)
        except ValueError as exc:
            raise ValueError("Absolute image_path must be inside data_dir") from exc
        if relative != expected_relative:
            raise ValueError(f"Image path/scene mismatch: {row['id']}")
    elif len(image_ref.parts) > 1 and image_ref != expected_relative:
        raise ValueError(f"Expected a filename or interval5_<scene>/interval5_CAM path: {row['id']}")
    rgb = (data_root / expected_relative).resolve()
    root = Path(depth_root).resolve()
    candidates = [root / row["scene"] / (image_ref.stem + suffix) for suffix in (".npy", ".npz")]
    depth = next((path for path in candidates if path.is_file()), candidates[0]).resolve()
    if not rgb.is_relative_to(data_root) or not depth.is_relative_to(root):
        raise ValueError(f"Asset resolves outside its root: {row['id']}")
    if not rgb.is_file() or not depth.is_file():
        raise FileNotFoundError(f"Missing RGB or DRoPE predicted depth: {row['id']} ({rgb}, {depth})")
    return str(rgb), str(depth)


def write_atomic(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".uavscenes-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if jsonl:
                for row in value:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            else:
                json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_generator_bundle(model_path, adapter_path, max_new_tokens):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from train_depth_rope_qwen3vl import attach_uav_depth_rope, get_qwen3vl_core_model
    from tcot.qwen3vl_runtime import GeneratorBundle

    config = json.loads((Path(adapter_path) / "uav_depth_rope_config.json").read_text(encoding="utf-8"))
    if (config.get("experiment") != "c_estimated_qwen_vit_depth_rope"
            or config.get("depth_rope_layout") != "balanced"
            or config.get("depth_rope_mode", "replace") != "replace"):
        raise ValueError("UAVScenes requires the balanced/replace estimated-depth DRoPE adapter")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    num_bins = int(config["depth_num_bins"])
    if num_bins < 2:
        raise ValueError("depth_num_bins must be at least 2")
    model = attach_uav_depth_rope(
        model, experiment=config["experiment"], depth_num_bins=num_bins,
        depth_rope_dim=int(config["depth_rope_dim"]),
        depth_rope_init_scale=float(config.get("depth_rope_init_scale", 0.)),
        depth_rope_theta=float(config.get("depth_rope_theta", 10000.)),
    )
    weights = torch.load(Path(adapter_path) / "uav_depth_rope.pt", map_location="cpu", weights_only=True)
    if "gate_active" in weights and float(weights["gate_active"]) != 0.:
        raise ValueError("Active soft-mode gate is incompatible with replace inference")
    weights = {key: value for key, value in weights.items() if key not in {"gate_raw", "gate_active"}}
    state = get_qwen3vl_core_model(model).uav_depth_rope_state
    state.load_state_dict(weights)
    state.to(device=next(model.parameters()).device, dtype=torch.float32)
    if not bool(torch.isfinite(state.scale).all()) or float(state.scale) == 0.:
        raise ValueError("Checkpoint depth phase scale must be finite and nonzero")
    model.eval()
    return GeneratorBundle(model, processor, config["experiment"], num_bins, max_new_tokens)


def inference_parser(description, tcot=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--qa_file", default=str(Path(__file__).resolve().parent / "benchmarks/uavscenes_metric_v2.json"))
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--estimated_depth_root", required=True, help="DRoPE predicted depth, <root>/<scene>/<stem>.npy or .npz")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512 if tcot else 8)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check_only", action="store_true", help="Check benchmark and input assets without loading a model")
    if tcot:
        parser.add_argument("--tool_tokens", type=int, default=192)
        parser.add_argument("--bbox_file", required=True, help="Predicted top-1 bbox JSONL; not QA GT boxes")
        parser.add_argument("--tool_depth_root", required=True, help="Metric-scale predicted tool depth, independent of DRoPE depth")
        parser.add_argument("--intrinsics_root", required=True, help="Predicted pixel intrinsics, <root>/<scene>/<stem>.K.npy")
    return parser


def run_inference(args, method, runner, backend_factory=None):
    if min(args.max_new_tokens, args.save_every, getattr(args, "tool_tokens", 1)) < 1:
        raise ValueError("Token budgets and save interval must be positive")
    rows = load_benchmark(args.qa_file, args.max_samples)
    output = Path(args.output)
    inputs = {Path(args.qa_file).resolve()}
    if backend_factory:
        inputs.add(Path(args.bbox_file).resolve())
    if output.resolve() in inputs:
        raise ValueError("Output must not overwrite benchmark or perception assets")
    assets = {row["id"]: resolve_assets(row, args.data_dir, args.estimated_depth_root) for row in rows}
    backend = backend_factory(args) if backend_factory else None
    print(f"Selected {len(rows)} UAVScenes samples; RGB/DRoPE depth files verified", flush=True)
    if args.check_only:
        if backend:
            print("Tool asset status:", backend.check_assets(rows), flush=True)
        return
    config = dict(method=method, model_path=str(Path(args.model_path).resolve()),
                  adapter_path=str(Path(args.adapter_path).resolve()), data_dir=str(Path(args.data_dir).resolve()),
                  estimated_depth_root=str(Path(args.estimated_depth_root).resolve()), max_new_tokens=args.max_new_tokens,
                  depth_norm_mode="inverse", invalid_depth_threshold=65000., lower_percentile=1., upper_percentile=95.,
                  do_sample=False, protocol="universal_prior_no_gate" if backend else "drope_mcq",
                  answer_extractor="cot" if backend else "legacy-direct")
    if backend:
        import hashlib
        from dronepur_common import file_digest

        root = Path(__file__).resolve().parent
        files = {Path(path).resolve() for paths in assets.values() for path in paths}
        for row in rows:
            stem = Path(row["image_path"].replace("\\", "/")).stem
            files.update(backend.depth_root / row["scene"] / (stem + suffix) for suffix in (".npy", ".npz"))
            files.add(backend.intrinsics_root / row["scene"] / (stem + ".K.npy"))
        fingerprints = {str(path): file_digest(path) if path.is_file() else "missing" for path in sorted(files)}
        code = [Path(__file__).resolve(), root / "inference/inference_tcot_uavscenes.py",
                root / "drope_common.py", root / "train_depth_rope_qwen3vl.py"]
        code += sorted((root / "tcot").rglob("*.py"))
        config.update(tool_tokens=args.tool_tokens, bbox_file=str(Path(args.bbox_file).resolve()),
                      bbox_sha256=file_digest(args.bbox_file), qa_sha256=file_digest(args.qa_file),
                      assets_sha256=hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest(),
                      code_sha256={str(path.relative_to(root)): file_digest(path) for path in code},
                      tool_depth_root=str(Path(args.tool_depth_root).resolve()), intrinsics_root=str(Path(args.intrinsics_root).resolve()),
                      object_binding="id_prebound_predicted_bboxes")
    selected = {row["id"]: row for row in rows}
    completed = {}
    if output.exists():
        if not args.resume:
            raise FileExistsError("Output exists; use --resume or a new output file")
        for record in read_predictions([output]):
            sid = record["id"]
            if sid not in selected:
                raise ValueError(f"Saved ID outside selected benchmark: {sid}")
            row = selected[sid]
            if (any(record.get(key) != row[key] for key in ("task", "subset", "scene", "question", "choices", "image_path"))
                    or record.get("gt_letter") != row["answer_letter"] or record.get("run_config") != config
                    or record.get("depth_path") != assets[sid][1] or not isinstance(record.get("response"), str)):
                raise ValueError(f"Saved record does not match this run: {sid}")
            completed[sid] = record
    pending = [row for row in rows if row["id"] not in completed]
    if pending:
        bundle = load_generator_bundle(args.model_path, args.adapter_path, args.max_new_tokens)
        try:
            for index, row in enumerate(pending, 1):
                from PIL import Image
                rgb_path, depth_path = assets[row["id"]]
                with Image.open(rgb_path) as handle:
                    image = handle.convert("RGB")
                record = runner(bundle, image, row, depth_path, args, backend)
                record.update(id=row["id"], task=row["task"], subset=row["subset"], scene=row["scene"],
                              question=row["question"], choices=row["choices"], image_path=row["image_path"],
                              depth_path=depth_path, gt_letter=row["answer_letter"], method=method,
                              uav_depth_experiment=bundle.experiment, run_config=config)
                completed[row["id"]] = record
                print(f"[{len(completed)}/{len(rows)}] {row['id']}", flush=True)
                if index % args.save_every == 0:
                    write_atomic(output, [completed[r["id"]] for r in rows if r["id"] in completed], jsonl=True)
        finally:
            write_atomic(output, [completed[r["id"]] for r in rows if r["id"] in completed], jsonl=True)
    else:
        write_atomic(output, [completed[r["id"]] for r in rows], jsonl=True)
    print(f"Saved {len(completed)} records to {output}")
