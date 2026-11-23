from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GeneratorBundle:
    model: object
    processor: object
    experiment: str
    depth_num_bins: int
    max_new_tokens: int = 512


def load_uav_depth_rope(model, adapter_path):
    import torch
    from train_depth_rope_qwen3vl import (
        EXPERIMENT_ESTIMATED, EXPERIMENTS,
        attach_uav_depth_rope, get_qwen3vl_core_model,
    )

    adapter_path = Path(adapter_path)
    with (adapter_path / "uav_depth_rope_config.json").open() as handle:
        config = json.load(handle)
    experiment = config.get("experiment") or EXPERIMENT_ESTIMATED
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unsupported experiment: {experiment}")
    if config.get("depth_rope_layout", "balanced") != "balanced" or config.get("depth_rope_mode", "replace") != "replace":
        raise ValueError("This historical Qwen3 pipeline requires balanced replace DRoPE")
    model = attach_uav_depth_rope(
        model, experiment=experiment,
        depth_num_bins=int(config["depth_num_bins"]),
        depth_rope_dim=int(config["depth_rope_dim"]),
        depth_rope_init_scale=float(config.get("depth_rope_init_scale", 0.)),
        depth_rope_theta=float(config.get("depth_rope_theta", 10000.)),
    )
    state = get_qwen3vl_core_model(model).uav_depth_rope_state
    state.load_state_dict(torch.load(adapter_path / "uav_depth_rope.pt", map_location="cpu", weights_only=True))
    state.to(device=next(model.parameters()).device, dtype=torch.float32)
    return model, experiment, int(config["depth_num_bins"])


def load_generator_bundle(model_path, adapter_path, max_new_tokens=512):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
    except OSError:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model, experiment, num_bins = load_uav_depth_rope(model, adapter_path)
    model.eval()
    return GeneratorBundle(model, processor, experiment, num_bins, max_new_tokens)


def make_messages(image, prompt, system_prompt="You are a helpful assistant."):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
    ]


def append_tool_history(messages, tool_pick, prompt):
    return messages + [
        {"role": "assistant", "content": [{"type": "text", "text": tool_pick}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def generate_text(bundle, messages, depth_path, max_new_tokens=None):
    import torch
    from train_depth_rope_qwen3vl import build_depth_rope_bins_for_grid, get_qwen3vl_core_model

    with torch.inference_mode():
        inputs = bundle.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(bundle.model.device)
        grids = inputs.get("image_grid_thw")
        if grids is None or len(grids) != 1:
            raise ValueError("DRoPE supports exactly one image per generation")
        bins = build_depth_rope_bins_for_grid(
            str(depth_path), grids[0].cpu(), num_bins=bundle.depth_num_bins,
            invalid_depth_threshold=65000., lower_percentile=1., upper_percentile=95.,
            depth_norm_mode="inverse",
        ).to(bundle.model.device)
        core = get_qwen3vl_core_model(bundle.model)
        core.visual._pending_depth_rope_bins = bins
        try:
            output = bundle.model.generate(
                **inputs, do_sample=False,
                max_new_tokens=bundle.max_new_tokens if max_new_tokens is None else max_new_tokens,
            )
        finally:
            core.visual._pending_depth_rope_bins = None
        return bundle.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def generate_independent_answer(bundle, image, question, depth_path):
    from drope_common import build_drope_messages

    messages = build_drope_messages(image, question, bundle.experiment)
    return generate_text(bundle, messages, depth_path, max_new_tokens=128)
