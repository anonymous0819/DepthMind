"""Qwen3-VL DRoPE training with fixed balanced depth-frequency allocation.

Binned inverse depth supplies rotary phases equally to the low-frequency
positions of the two spatial axes. The depth phase scale is warmed up during
LoRA training; RGB token features and the visual sequence length are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import types
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast

from drope_common import (
    SYSTEM_PROMPT,
    build_depth_path,
    build_image_path,
    DEPTH_NORM_MODE_INVERSE,
    DEPTH_NORM_MODES,
    get_valid_depth_mask,
    load_depth,
    normalize_depth,
    robust_depth_bounds,
    build_estimated_depth_path,
)


IGNORE_INDEX = -100

EXPERIMENT_ESTIMATED = "c_estimated_qwen_vit_depth_rope"
EXPERIMENT_GT = "c_qwen_vit_depth_rope"
EXPERIMENTS = (EXPERIMENT_ESTIMATED, EXPERIMENT_GT)



# ---------------------------------------------------------------------------
# Depth-bin utilities (patch-level, aligned to Qwen3VL vision full-resolution
# tokens, i.e. the sequence used by ``rot_pos_emb`` *before* the merger).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _load_and_bin_depth(
    depth_path: str,
    invalid_depth_threshold: float,
    lower_percentile: float,
    upper_percentile: float,
    depth_norm_mode: str,
) -> np.ndarray:
    """Load a depth map and return its per-pixel bin-space value in [0, 1].

    The returned array uses the inverse-depth convention: near pixels map to
    small values, far pixels map to large values, and invalid pixels are set
    to a sentinel value (=1.0 corresponds to "far / invalid" for downstream
    binning).
    """

    depth = load_depth(depth_path)
    valid_mask = get_valid_depth_mask(depth, invalid_depth_threshold)
    lower, upper = robust_depth_bounds(depth, valid_mask, lower_percentile, upper_percentile)
    # normalize_depth returns (1.0 - inverse_depth_min_max) with far=1.0.
    normalized = normalize_depth(
        depth,
        valid_mask,
        lower,
        upper,
        use_log_depth=False,
        mode=depth_norm_mode,
    )
    # Flip so near=1.0, far=0.0 (matches "inverse depth" intuition).
    return (1.0 - normalized).astype(np.float32)


def build_depth_rope_bins_for_grid(
    depth_path: str,
    image_grid_thw: torch.Tensor,
    num_bins: int,
    invalid_depth_threshold: float,
    lower_percentile: float,
    upper_percentile: float,
    depth_norm_mode: str = DEPTH_NORM_MODE_INVERSE,
) -> torch.Tensor:
    """Return per-patch depth bin ids at the Qwen3VL vision full-resolution
    grid (i.e. the sequence used by ``rot_pos_emb`` before the merger).

    Output shape: ``(grid_t * grid_h * grid_w,)`` with int64 values in
    ``[0, num_bins)``.
    """

    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")

    grid_t = int(image_grid_thw[0].item())
    grid_h = int(image_grid_thw[1].item())
    grid_w = int(image_grid_thw[2].item())

    inv_map = _load_and_bin_depth(
        depth_path,
        invalid_depth_threshold,
        lower_percentile,
        upper_percentile,
        depth_norm_mode,
    )
    height, width = inv_map.shape

    y_bins = np.linspace(0, height, grid_h + 1).round().astype(np.int64)
    x_bins = np.linspace(0, width, grid_w + 1).round().astype(np.int64)

    patch_values = np.zeros((grid_h, grid_w), dtype=np.float32)
    for y_idx in range(grid_h):
        y0, y1 = int(y_bins[y_idx]), int(y_bins[y_idx + 1])
        if y1 <= y0:
            y1 = min(height, y0 + 1)
        for x_idx in range(grid_w):
            x0, x1 = int(x_bins[x_idx]), int(x_bins[x_idx + 1])
            if x1 <= x0:
                x1 = min(width, x0 + 1)
            cell = inv_map[y0:y1, x0:x1]
            if cell.size == 0:
                patch_values[y_idx, x_idx] = 0.0
            else:
                patch_values[y_idx, x_idx] = float(cell.mean())

    patch_values = np.clip(patch_values, 0.0, 1.0)
    bin_ids = np.clip(
        np.floor(patch_values * num_bins).astype(np.int64),
        0,
        num_bins - 1,
    )
    # IMPORTANT: Qwen ViT's ``rot_pos_emb`` produces positional embeddings in
    # "block-major then intra-block row-major" order (i.e. it iterates 2x2
    # spatial-merge blocks first, then walks each block's 4 patches). A naive
    # ``bin_ids.reshape(-1)`` would be plain full-image row-major, which
    # mis-aligns depth bins vs vanilla RoPE positions (66% of positions swap
    # inside every 2x2 block). We instead permute (H, W) -> (H/2, W/2, 2, 2)
    # -> flatten so ``bin_ids[k]`` corresponds *exactly* to the same patch as
    # ``rotary_pos_emb[k]``. This applies to both Qwen3-VL and Qwen2.5-VL
    # (both use spatial_merge_size=2).
    merge_size = 2
    if grid_h % merge_size == 0 and grid_w % merge_size == 0:
        bin_ids = (
            bin_ids
            .reshape(grid_h // merge_size, merge_size,
                     grid_w // merge_size, merge_size)
            .transpose(0, 2, 1, 3)
            .reshape(-1)
        )
    else:
        # Fallback: shouldn't happen for standard Qwen preprocessing which
        # always pads/resizes grid_h and grid_w to multiples of merge_size.
        bin_ids = bin_ids.reshape(-1)
    if grid_t > 1:
        bin_ids = np.tile(bin_ids, grid_t)
    return torch.from_numpy(bin_ids)


# ---------------------------------------------------------------------------
# Prompt / dataset
# ---------------------------------------------------------------------------


DEPTH_ROPE_PROMPT_LINES = (
    "The image is an RGB image of an urban UAV scene.",
    (
        "The vision transformer receives an additional depth-aware rotary "
        "position signal derived from the aligned depth map, so that two "
        "patches with different metric depths are treated as farther apart "
        "in the attention geometry."
    ),
    (
        "Use this depth-aware attention structure for distance, size, "
        "front/behind, left/right, and clock-direction reasoning."
    ),
)


def build_experiment_prompt(question: str, experiment: str) -> str:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unsupported experiment: {experiment}")
    return "\n".join(list(DEPTH_ROPE_PROMPT_LINES) + [f"Question: {question}"])


class Open3DVQADepthRopeDataset(Dataset):
    def __init__(
        self,
        qa_file: str,
        data_dir: str,
        experiment: str,
        max_samples: int | None = None,
        seed: int = 42,
        estimated_depth_root: str | None = None,
    ):
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unsupported experiment: {experiment}")
        self.qa_file = qa_file
        self.data_dir = data_dir
        self.experiment = experiment
        self.seed = seed
        self.estimated_depth_root = estimated_depth_root
        self.samples = self._load_samples(qa_file)
        if max_samples is not None and max_samples > 0 and max_samples < len(self.samples):
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, max_samples)

    def _load_samples(self, qa_file: str) -> list[dict[str, Any]]:
        with open(qa_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        samples: list[dict[str, Any]] = []
        for dataset, scenes in data.items():
            for scene, entries in scenes.items():
                for item in entries:
                    question = item.get("query_question") or ""
                    conversation = item.get("conversation") or []
                    answer = conversation[1].get("value", "") if len(conversation) > 1 else ""
                    image_info = item.get("image_info") or {}
                    if question and answer and image_info.get("image_path"):
                        samples.append(
                            {
                                "id": item.get("id"),
                                "dataset": dataset,
                                "scene": scene,
                                "image_info": image_info,
                                "question": question.strip(),
                                "answer": answer.strip(),
                            }
                        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        dataset = sample["dataset"]
        scene = sample["scene"]
        image_info = sample["image_info"]

        image_path = build_image_path(self.data_dir, dataset, scene, image_info)
        if self.estimated_depth_root:
            depth_path = build_estimated_depth_path(
                self.estimated_depth_root,
                self.data_dir,
                dataset,
                scene,
                image_info,
            )
        else:
            depth_path = build_depth_path(self.data_dir, dataset, scene, image_info)
        if depth_path is None:
            raise FileNotFoundError(f"Depth path cannot be inferred for sample {sample.get('id')}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(depth_path):
            depth_source = "estimated" if self.estimated_depth_root else "ground-truth"
            raise FileNotFoundError(f"{depth_source.capitalize()} depth file not found: {depth_path}")

        image = Image.open(image_path).convert("RGB")
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": build_experiment_prompt(sample["question"], self.experiment)},
        ]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
        ]
        return {
            "id": sample.get("id"),
            "depth_path": depth_path,
            "messages": messages,
            "prompt_messages": messages[:-1],
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


@dataclass
class Qwen3VLDepthRopeCollator:
    processor: Any
    spatial_merge_size: int
    depth_num_bins: int
    depth_lower_percentile: float
    depth_upper_percentile: float
    invalid_depth_threshold: float
    depth_norm_mode: str = DEPTH_NORM_MODE_INVERSE

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_inputs = []
        for feature in features:
            full_inputs = self.processor.apply_chat_template(
                feature["messages"],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_inputs = self.processor.apply_chat_template(
                feature["prompt_messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            squeezed_inputs = {}
            for key, value in full_inputs.items():
                if key in {"input_ids", "attention_mask"}:
                    squeezed_inputs[key] = value.squeeze(0)
                else:
                    squeezed_inputs[key] = value

            prompt_len = int(prompt_inputs["input_ids"].shape[-1])
            labels = squeezed_inputs["input_ids"].clone()
            labels[:prompt_len] = IGNORE_INDEX
            squeezed_inputs["labels"] = labels

            image_grid_thw = squeezed_inputs.get("image_grid_thw")
            if image_grid_thw is None:
                raise ValueError("Qwen3-VL processor did not return image_grid_thw.")
            if int(image_grid_thw.shape[0]) != 1:
                raise ValueError(
                    "DepthRope collator currently supports exactly one image per sample, "
                    f"got image_grid_thw with shape {image_grid_thw.shape}."
                )

            depth_bins = build_depth_rope_bins_for_grid(
                feature["depth_path"],
                image_grid_thw[0],
                num_bins=self.depth_num_bins,
                invalid_depth_threshold=self.invalid_depth_threshold,
                lower_percentile=self.depth_lower_percentile,
                upper_percentile=self.depth_upper_percentile,
                depth_norm_mode=self.depth_norm_mode,
            )
            expected_len = int(image_grid_thw.prod(-1).sum().item())
            if int(depth_bins.shape[0]) != expected_len:
                raise ValueError(
                    f"Depth-rope bins length {depth_bins.shape[0]} does not "
                    f"match vision full-resolution token count {expected_len} "
                    f"for {feature['depth_path']}."
                )
            squeezed_inputs["uav_depth_rope_bins_per_sample"] = depth_bins

            batch_inputs.append(squeezed_inputs)

        tokenizer = self.processor.tokenizer
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

        max_length = max(item["input_ids"].shape[0] for item in batch_inputs)
        input_ids = []
        attention_mask = []
        labels = []
        padding_side = getattr(tokenizer, "padding_side", "right")
        for item in batch_inputs:
            seq_len = item["input_ids"].shape[0]
            pad_len = max_length - seq_len
            input_pad = torch.full((pad_len,), pad_token_id, dtype=item["input_ids"].dtype)
            mask_pad = torch.zeros((pad_len,), dtype=item["attention_mask"].dtype)
            label_pad = torch.full((pad_len,), IGNORE_INDEX, dtype=item["labels"].dtype)
            if padding_side == "left":
                input_ids.append(torch.cat([input_pad, item["input_ids"]], dim=0))
                attention_mask.append(torch.cat([mask_pad, item["attention_mask"]], dim=0))
                labels.append(torch.cat([label_pad, item["labels"]], dim=0))
            else:
                input_ids.append(torch.cat([item["input_ids"], input_pad], dim=0))
                attention_mask.append(torch.cat([item["attention_mask"], mask_pad], dim=0))
                labels.append(torch.cat([item["labels"], label_pad], dim=0))

        batch = {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }
        if all("pixel_values" in item for item in batch_inputs):
            batch["pixel_values"] = torch.cat([item["pixel_values"] for item in batch_inputs], dim=0)
        if all("image_grid_thw" in item for item in batch_inputs):
            batch["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch_inputs], dim=0)

        # Depth bins are stored as a flat concatenation along the vision
        # sequence dimension (which is exactly how Qwen3VLVisionModel receives
        # ``hidden_states``). A parallel ``sample_offsets`` tensor is unnecessary
        # here because ``grid_thw`` already encodes per-image sequence lengths.
        depth_rope_bins = torch.cat(
            [item["uav_depth_rope_bins_per_sample"] for item in batch_inputs],
            dim=0,
        )
        batch["uav_depth_rope_bins"] = depth_rope_bins
        return batch


# ---------------------------------------------------------------------------
# Depth-RoPE injection into Qwen3VLVisionModel
# ---------------------------------------------------------------------------


class DepthRopeState(nn.Module):
    """Depth phase table and global scale for balanced spatial allocation."""

    def __init__(
        self,
        num_bins: int,
        depth_rope_dim: int,
        theta: float = 10000.0,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if num_bins <= 0 or theta <= 0:
            raise ValueError("num_bins and theta must be positive")
        if depth_rope_dim <= 0 or depth_rope_dim % 4 != 0:
            raise ValueError("depth_rope_dim must be a positive multiple of 4")
        self.num_bins = int(num_bins)
        self.depth_rope_dim = int(depth_rope_dim)
        self.theta = float(theta)
        inv_freq = 1.0 / (theta ** (torch.arange(0, depth_rope_dim, 2, dtype=torch.float32) / depth_rope_dim))
        bin_positions = torch.arange(num_bins, dtype=torch.float32)
        self.register_buffer("freq_table", torch.outer(bin_positions, inv_freq), persistent=True)
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def get_freqs(self, bins: torch.Tensor) -> torch.Tensor:
        table = self.freq_table.to(device=bins.device)
        picked = table[bins.long()]
        return picked * self.scale.to(device=bins.device, dtype=picked.dtype)


def _make_patched_rot_pos_emb(original_rot_pos_emb, depth_state: DepthRopeState):
    depth_rope_half = depth_state.depth_rope_dim // 2

    def patched_rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        # ``self._pending_depth_rope_bins`` is set by the patched forward
        # right before it calls ``rot_pos_emb``.
        vanilla = original_rot_pos_emb(grid_thw)
        depth_bins = getattr(self, "_pending_depth_rope_bins", None)
        if depth_bins is None:
            return vanilla

        # Vanilla shape is (seq_len, dim); ``dim`` == vision head_dim / 2
        # (Qwen3-VL vision uses [row(dim/2) | col(dim/2)] concat layout).
        if vanilla.dim() != 2:
            raise ValueError(f"Unexpected vanilla rot_pos_emb shape: {vanilla.shape}")
        seq_len, dim = vanilla.shape
        if int(depth_bins.shape[0]) != seq_len:
            raise ValueError(
                f"depth_rope_bins length {depth_bins.shape[0]} does not match "
                f"vision sequence length {seq_len}."
            )
        if depth_rope_half > dim // 2:
            raise ValueError(
                f"depth_rope_half={depth_rope_half} exceeds per-channel width dim/2={dim // 2}. "
                f"Cannot fit depth into a single row/col channel."
            )
        depth_freqs = depth_state.get_freqs(depth_bins.to(device=vanilla.device))
        depth_freqs = depth_freqs.to(dtype=vanilla.dtype)
        patched = vanilla.clone()
        half_dim = dim // 2
        q = depth_rope_half // 2
        patched[:, half_dim - q:half_dim] = depth_freqs[:, :q]
        patched[:, dim - q:dim] = depth_freqs[:, q:]
        return patched

    return patched_rot_pos_emb


def patched_qwen3vl_model_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Any = None,
    inputs_embeds: torch.FloatTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    uav_depth_rope_bins: torch.Tensor | None = None,
    **kwargs: Any,
) -> Qwen3VLModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None
    deepstack_image_embeds = None
    deepstack_video_embeds = None

    if pixel_values is not None:
        # ``get_image_features`` (from the vanilla Qwen3-VL) does *not* accept
        # extra kwargs and does *not* forward kwargs to ``self.visual``. So
        # instead of routing depth bins through kwargs, we directly stash them
        # on the visual submodule before calling ``get_image_features``; our
        # patched ``rot_pos_emb`` picks them up from there. We always clear
        # the slot afterwards to avoid state leaking between forward calls.
        try:
            if uav_depth_rope_bins is not None:
                self.visual._pending_depth_rope_bins = uav_depth_rope_bins.to(device=pixel_values.device)
            image_embeds, deepstack_image_embeds = self.get_image_features(
                pixel_values,
                image_grid_thw,
            )
        finally:
            self.visual._pending_depth_rope_bins = None
        image_embeds = torch.cat(list(image_embeds), dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        past_seen_tokens = 0
        if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            past_seen_tokens = past_key_values.get_seq_length()
        prefill_stage = (
            (cache_position is not None and int(cache_position[0].item()) == 0)
            or past_key_values is None
            or past_seen_tokens == 0
        )
        if prefill_stage or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )


def get_qwen3vl_core_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        conditional_model = model.base_model.model
    else:
        conditional_model = model
    if not hasattr(conditional_model, "model"):
        raise ValueError("Could not locate inner Qwen3VLModel from the provided model.")
    return conditional_model.model


def attach_uav_depth_rope(
    peft_model: nn.Module,
    experiment: str,
    depth_num_bins: int,
    depth_rope_dim: int,
    depth_rope_init_scale: float,
    depth_rope_theta: float,
) -> nn.Module:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unsupported experiment: {experiment}")
    core_model = get_qwen3vl_core_model(peft_model)

    vision = core_model.visual
    depth_state = DepthRopeState(
        num_bins=depth_num_bins,
        depth_rope_dim=depth_rope_dim,
        theta=depth_rope_theta,
        init_scale=depth_rope_init_scale,
    )
    device = peft_model.get_input_embeddings().weight.device
    depth_state.to(device=device, dtype=torch.float32)

    # Store the depth state module on the *core model* so that our save
    # callback can dump it. We also stash a reference on the visual submodule
    # so its patched methods can access it without going up the tree.
    core_model.uav_depth_rope_state = depth_state
    vision._uav_depth_rope_state = depth_state
    vision._pending_depth_rope_bins = None

    # We only patch ``rot_pos_emb``: it picks up ``vision._pending_depth_rope_bins``
    # which is set by ``patched_qwen3vl_model_forward`` right before every
    # ``get_image_features`` call. We do NOT patch ``vision.forward`` because
    # a no-op wrapper would just add overhead.
    original_rot_pos_emb = types.MethodType(type(vision).rot_pos_emb, vision)
    vision.rot_pos_emb = types.MethodType(
        _make_patched_rot_pos_emb(original_rot_pos_emb, depth_state),
        vision,
    )

    core_model.forward = types.MethodType(patched_qwen3vl_model_forward, core_model)

    core_model.uav_depth_rope_experiment = experiment
    core_model.uav_depth_rope_config = {
        "experiment": experiment,
        "depth_num_bins": depth_num_bins,
        "depth_rope_dim": depth_rope_dim,
        "depth_rope_init_scale": depth_rope_init_scale,
        "depth_rope_theta": depth_rope_theta,
        "depth_rope_layout": "balanced",
    }
    return peft_model


def save_uav_depth_rope(model: nn.Module, output_dir: str) -> None:
    core_model = get_qwen3vl_core_model(model)
    if not hasattr(core_model, "uav_depth_rope_state"):
        return
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        core_model.uav_depth_rope_state.state_dict(),
        os.path.join(output_dir, "uav_depth_rope.pt"),
    )
    with open(os.path.join(output_dir, "uav_depth_rope_config.json"), "w", encoding="utf-8") as file:
        json.dump(core_model.uav_depth_rope_config, file, ensure_ascii=False, indent=2)


class UAVDepthRopeSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step is None:
            return control
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_uav_depth_rope(model, checkpoint_dir)
        return control


class DepthRopeScaleScheduler(TrainerCallback):
    """Ramp the depth-RoPE scale from 0.0 to ``target`` over the first
    ``warmup_steps`` optimiser steps. If ``warmup_steps <= 0`` the scale is
    set to ``target`` from the start.
    """

    def __init__(self, warmup_steps: int, target: float) -> None:
        self.warmup_steps = int(warmup_steps)
        self.target = float(target)

    def _set_scale(self, model: nn.Module, value: float) -> None:
        core_model = get_qwen3vl_core_model(model)
        depth_state = getattr(core_model, "uav_depth_rope_state", None)
        if depth_state is None:
            return
        with torch.no_grad():
            depth_state.scale.fill_(value)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        initial = 0.0 if self.warmup_steps > 0 else self.target
        self._set_scale(model, initial)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.warmup_steps <= 0:
            return control
        progress = min(1.0, float(state.global_step + 1) / float(self.warmup_steps))
        self._set_scale(model, self.target * progress)
        return control


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_lora_targets(targets: str) -> list[str]:
    return [target.strip() for target in targets.split(",") if target.strip()]


def build_dataset(args: argparse.Namespace, qa_file: str, max_samples: int | None) -> Open3DVQADepthRopeDataset:
    return Open3DVQADepthRopeDataset(
        qa_file=qa_file,
        data_dir=args.data_dir,
        experiment=args.experiment,
        max_samples=max_samples,
        seed=args.seed,
        estimated_depth_root=args.estimated_depth_root,
    )


def print_trainable_parameters(model: nn.Module) -> None:
    trainable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    percentage = 100 * trainable_params / total_params if total_params > 0 else 0.0
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.4f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA/QLoRA fine-tuning for Qwen3-VL with Depth-RoPE injection into the vision transformer."
    )
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=EXPERIMENT_ESTIMATED)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_qa_file", required=True)
    parser.add_argument("--val_qa_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--estimated_depth_root",
        default=None,
        help=(
            "Root directory of estimated depth maps (e.g. DA3 exports). If set, the "
            "dataset reads estimated depth instead of the GT depth aligned to the RGB."
        ),
    )

    parser.add_argument("--depth_lower_percentile", type=float, default=1.0)
    parser.add_argument("--depth_upper_percentile", type=float, default=95.0)
    parser.add_argument("--invalid_depth_threshold", type=float, default=65000.0)
    parser.add_argument(
        "--depth_norm_mode",
        choices=DEPTH_NORM_MODES,
        default=DEPTH_NORM_MODE_INVERSE,
        help=(
            "Depth normalization mode used when computing per-patch bins. "
            "Inverse-depth is unit-invariant across the four subsets and is the recommended default."
        ),
    )

    parser.add_argument("--depth_num_bins", type=int, default=128)
    parser.add_argument("--depth_rope_dim", type=int, default=16, help="Positive multiple of 4. Rotary dimensions allocated to depth.")
    parser.add_argument("--depth_rope_theta", type=float, default=10000.0)
    parser.add_argument("--depth_rope_target_scale", type=float, default=1.0, help="Depth phase scale after warmup.")
    parser.add_argument("--depth_rope_warmup_steps", type=int, default=500, help="Steps to ramp the depth phase scale to its target. Set 0 to disable warmup.")

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--no_eval", action="store_true", help="Disable Trainer eval_loss during training.")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--report_to", default="none")
    args = parser.parse_args()

    if args.experiment == EXPERIMENT_ESTIMATED and not args.estimated_depth_root:
        raise ValueError(
            f"Experiment {args.experiment} requires --estimated_depth_root to point to the DA3 (or other estimated) depth exports."
        )

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    train_dataset = build_dataset(args, args.train_qa_file, args.max_train_samples)
    eval_dataset = None if args.no_eval else build_dataset(args, args.val_qa_file, args.max_eval_samples)

    print(f"Experiment: {args.experiment}")
    print(f"Depth source: {'estimated' if args.estimated_depth_root else 'ground-truth'}")
    print(f"Estimated depth root: {args.estimated_depth_root or 'N/A'}")
    print(f"Depth normalization mode: {args.depth_norm_mode}")
    print(f"Depth num_bins: {args.depth_num_bins}")
    print(f"Depth-RoPE dim: {args.depth_rope_dim}")
    print(f"Depth-RoPE theta: {args.depth_rope_theta}")
    print(f"Depth-RoPE warmup steps: {args.depth_rope_warmup_steps} -> target scale {args.depth_rope_target_scale}")
    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval samples: {len(eval_dataset)}")
    else:
        print("Eval disabled.")

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=parse_lora_targets(args.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)

    # Sanity: check depth_rope_dim <= vision head_dim.
    core_model_probe = get_qwen3vl_core_model(model)
    vision_head_dim = int(core_model_probe.visual.config.hidden_size) // int(core_model_probe.visual.config.num_heads)
    if args.depth_rope_dim > vision_head_dim:
        raise ValueError(
            f"depth_rope_dim={args.depth_rope_dim} exceeds vision head_dim={vision_head_dim}."
        )

    model = attach_uav_depth_rope(
        model,
        experiment=args.experiment,
        depth_num_bins=args.depth_num_bins,
        depth_rope_dim=args.depth_rope_dim,
        depth_rope_init_scale=0.0,
        depth_rope_theta=args.depth_rope_theta,
    )
    print_trainable_parameters(model)

    spatial_merge_size = int(model.base_model.model.config.vision_config.spatial_merge_size)
    data_collator = Qwen3VLDepthRopeCollator(
        processor=processor,
        spatial_merge_size=spatial_merge_size,
        depth_num_bins=args.depth_num_bins,
        depth_lower_percentile=args.depth_lower_percentile,
        depth_upper_percentile=args.depth_upper_percentile,
        invalid_depth_threshold=args.invalid_depth_threshold,
        depth_norm_mode=args.depth_norm_mode,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="no" if args.no_eval else "steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=args.report_to,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            UAVDepthRopeSaveCallback(),
            DepthRopeScaleScheduler(
                warmup_steps=args.depth_rope_warmup_steps,
                target=args.depth_rope_target_scale,
            ),
        ],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    save_uav_depth_rope(model, args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter, UAV depth-RoPE state, and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
