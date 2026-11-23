# DepthMind

This repository is the implementation of the DepthMind [DepthMind: Seeing and Thinking in 3D World from UAVs]

## Introduction

We focus on VLM spatial reasoning in UAV scenes and propose DepthMind to enable model to both see and think in 3D from UAV perspectives. DepthMind consists of two main components: DRoPE and TCoT. DRoPE exploits the directional geometric anisotropy to enocde the depth information into rotary coordinates via asymmetric frequency allocation, while TCoT combines VLM's intrinsic knowledge with evidence from external tools to resolve spatial and scale ambiguities. Extensive experiments demonstrate the effectiveness of DepthMind in UAV scenes.

## Directory Structure

```text
DepthMind/
├── train_depth_rope_qwen3vl.py
├── inference/
│   ├── inference_drope_open3dvqa.py
│   ├── inference_tcot_open3dvqa.py
│   ├── inference_drope_uavscenes.py
│   └── inference_tcot_uavscenes.py
├── evaluation/
│   ├── evaluate_open3dvqa.py
│   └── evaluate_uavscenes.py
├── uavscenes_common.py
├── drope_common.py
├── requirements.txt
├── tcot/
│   ├── cache/
│   │   ├── cached_geometry.py     # Open3DVQA cached assets and geometry
│   │   ├── uavscenes.py           # UAVScenes cached bbox/depth backend
│   │   ├── build_o3dvqa_perception_manifest.py
│   │   └── *_geometry.py          # shared relation/metric geometry
│   ├── metric_freeform.py
│   ├── metric_mcq.py
│   ├── fusion_stages.py           # Qwen3 relation chain
│   ├── o3dvqa_protocol.py          # universal prompts and validation
│   ├── question_prompts.py        #  metric MCQ reasoning prompts
│   ├── answer_prompts.py          # reasoning templates and option extraction
│   ├── tool_schema.py
│   ├── tool_selection.py
│   └── qwen3vl_runtime.py
├── external_tools/
│   ├── depth/
│   │   ├── extract_dronepur_da3_depth.py
│   │   ├── extract_o3dvqa_da3_depth.py
│   │   ├── extract_moge2_depth_o3dvqa.py
│   │   ├── extract_unidepth_depth_o3dvqa.py
│   │   └── metric_depth_extraction.py
│   ├── segmentation/
│   │   ├── extract_o3dvqa_sam_masks.py
│   │   └── sam_cue_helpers.py
│   ├── targets/
│   │   └── extract_targets_qwen3.py
│   ├── cues/                      # relation cache + optional 3D exports
│   │   ├── build_o3dvqa_relation_cues.py
│   │   ├── build_vertical3d_cues_o3dvqa.py
│   │   ├── build_lateral3d_cues_o3dvqa.py
│   │   └── build_clock3d_cues_o3dvqa.py
│   └── o3dvqa_extraction_common.py
├── benchmarks/
│   ├── uavscenes_metric.json
│   └── open3dvqa_metric_rank_random.json
└── scripts/                       # benchmark builders 
```


## Installation


```bash
python -m pip install -r requirements.txt
```


## Datasets
The Open3DVQA can be accessed [HERE](https://github.com/EmbodiedCity/Open3D-VQA.code).
The UAVScenes can be accessed [HERE](https://github.com/sijieaaa/UAVScenes).


```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export DATA_ROOT=/path/to/Open3DVQA
export DROPE_DEPTH_ROOT=/path/to/predicted_drope_depth
export ADAPTER_PATH="$PWD/outputs/drope_adapter"
export PERCEPTION_MANIFEST=/path/to/perception/manifest.json 
```

## Cache Preparation


### DRoPE: DA3 depth

```bash
export QA_FILE="$DATA_ROOT/test_qa.json"
export DA3_SRC=/path/to/Depth-Anything-3/src
export DA3_MODEL=/path/to/DA3NESTED-GIANT-LARGE-1.1

CUDA_VISIBLE_DEVICES=0 python external_tools/depth/extract_o3dvqa_da3_depth.py \
  --qa_file "$QA_FILE" \
  --data_dir "$DATA_ROOT" \
  --out_root "$DROPE_DEPTH_ROOT" \
  --da3_src "$DA3_SRC" \
  --da3_model "$DA3_MODEL" \
  --process_res 504 \
  --device cuda:0 \
  --skip_existing
```

### TCoT: predicted tool assets


```bash
export QA_FILE="$DATA_ROOT/test_qa.json"
export TOOL_CACHE_ROOT="$PWD/artifacts/perception"
export TARGET_FILE="$TOOL_CACHE_ROOT/targets.json"
export MASK_INDEX="$TOOL_CACHE_ROOT/masks.json"
export TOOL_DEPTH_ROOT="$TOOL_CACHE_ROOT/moge2_depth"
export PERCEPTION_MANIFEST="$TOOL_CACHE_ROOT/perception.json"
export RELATION_CUES="$TOOL_CACHE_ROOT/relation_cues_moge2.json"
export SAM3_SRC=/path/to/sam3
export SAM3_CHECKPOINT=/path/to/sam3_checkpoint.pt
export MOGE_SRC=/path/to/MoGe
export MOGE_CHECKPOINT=/path/to/moge2_checkpoint.pt

CUDA_VISIBLE_DEVICES=0 python external_tools/targets/extract_targets_qwen3.py \
  --qa_file "$QA_FILE" \
  --model_path "$MODEL_PATH" \
  --result_path "$TARGET_FILE" \
  --resume

CUDA_VISIBLE_DEVICES=0 python external_tools/segmentation/extract_o3dvqa_sam_masks.py \
  --target_file "$TARGET_FILE" \
  --data_dir "$DATA_ROOT" \
  --mask_dir "$TOOL_CACHE_ROOT/masks" \
  --result_path "$MASK_INDEX" \
  --sam3_src "$SAM3_SRC" \
  --sam3_checkpoint "$SAM3_CHECKPOINT" \
  --resume

CUDA_VISIBLE_DEVICES=0 python external_tools/depth/extract_moge2_depth_o3dvqa.py \
  --qa_file "$QA_FILE" \
  --data_dir "$DATA_ROOT" \
  --out_root "$TOOL_DEPTH_ROOT" \
  --moge_src "$MOGE_SRC" \
  --pretrained "$MOGE_CHECKPOINT" \
  --device cuda:0

python tcot/cache/build_o3dvqa_perception_manifest.py \
  --mask_file "$MASK_INDEX" \
  --data_dir "$DATA_ROOT" \
  --depth_root "$TOOL_DEPTH_ROOT" \
  --depth_backend moge2 \
  --depth_unit m \
  --output "$PERCEPTION_MANIFEST"

python external_tools/cues/build_o3dvqa_relation_cues.py \
  --target_file "$TARGET_FILE" \
  --mask_file "$MASK_INDEX" \
  --data_dir "$DATA_ROOT" \
  --depth_backend moge2 \
  --depth_root "$TOOL_DEPTH_ROOT" \
  --result_path "$RELATION_CUES" \
  --resume
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_depth_rope_qwen3vl.py \
  --experiment c_estimated_qwen_vit_depth_rope \
  --model_path "$MODEL_PATH" \
  --data_dir "$DATA_ROOT" \
  --train_qa_file /path/to/train_qa.json \
  --val_qa_file /path/to/val_qa.json \
  --estimated_depth_root "$DROPE_DEPTH_ROOT" \
  --output_dir "$ADAPTER_PATH" \
  --depth_norm_mode inverse \
  --depth_num_bins 128 \
  --depth_rope_dim 16 \
  --depth_rope_theta 10000 \
  --depth_rope_target_scale 1.0 \
  --depth_rope_warmup_steps 500 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4
```



## Open3DVQA inference

```bash
mkdir -p outputs
COMMON_ARGS=(
  --data_dir "$DATA_ROOT"
  --estimated_depth_root "$DROPE_DEPTH_ROOT"
  --model_path "$MODEL_PATH"
  --adapter_path "$ADAPTER_PATH"
  --perception_manifest "$PERCEPTION_MANIFEST"
)
```

### Relation

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_tcot_open3dvqa.py \
  "${COMMON_ARGS[@]}" \
  --module relation \
  --relation_cues "$RELATION_CUES" \
  --qa_file "$DATA_ROOT/test_qa.json" \
  --output outputs/tcot_relation_full_chain_online_draft.json
```

### Metric free-form

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_tcot_open3dvqa.py \
  "${COMMON_ARGS[@]}" \
  --module metric-freeform \
  --qa_file "$DATA_ROOT/test_qa.json" \
  --output outputs/tcot_metric_freeform.json
```

### Metric rank-random MCQ

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_tcot_open3dvqa.py \
  "${COMMON_ARGS[@]}" \
  --module metric-mcq \
  --qa_file benchmarks/open3dvqa_metric_rank_random.json \
  --output outputs/tcot_metric_rank_random.json
```


## Open3DVQA evaluation
Specify the judege model path
```bash
export JUDGE_MODEL_PATH="$MODEL_PATH"
```

### Free-form

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_open3dvqa.py \
  --mode freeform \
  --qa_file "$DATA_ROOT/test_qa.json" \
  --responses outputs/tcot_relation.json outputs/tcot_metric_freeform.json \
  --judge_model_path "$JUDGE_MODEL_PATH" \
  --output outputs/tcot_freeform_scores.json \
  --log_file outputs/tcot_freeform_scores.txt
```

### Hybrid

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_open3dvqa.py \
  --mode hybrid \
  --qa_file "$DATA_ROOT/test_qa.json" \
  --mcq_qa_file benchmarks/open3dvqa_metric_rank_random.json \
  --responses outputs/tcot_relation.json outputs/tcot_metric_rank_random.json \
  --judge_model_path "$JUDGE_MODEL_PATH" \
  --answer_extractor cot \
  --output outputs/tcot_hybrid_scores.json \
  --log_file outputs/tcot_hybrid_scores.txt
```

## UAVScenes


```bash
export UAVSCENES_DATA_ROOT=/path/to/UAVScenes
export UAVSCENES_DROPE_DEPTH_ROOT=/path/to/predicted_drope_depth
export UAVSCENES_TOOL_DEPTH_ROOT=/path/to/predicted_metric_depth
export UAVSCENES_INTRINSICS_ROOT=/path/to/predicted_intrinsics
export UAVSCENES_BBOX_FILE=/path/to/predicted_top1_bboxes.jsonl
```

### Inference

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_tcot_uavscenes.py \
  --data_dir "$UAVSCENES_DATA_ROOT" \
  --estimated_depth_root "$UAVSCENES_DROPE_DEPTH_ROOT" \
  --model_path "$MODEL_PATH" \
  --adapter_path "$ADAPTER_PATH" \
  --bbox_file "$UAVSCENES_BBOX_FILE" \
  --tool_depth_root "$UAVSCENES_TOOL_DEPTH_ROOT" \
  --intrinsics_root "$UAVSCENES_INTRINSICS_ROOT" \
  --output outputs/tcot_uavscenes.jsonl
```

### Evaluation

```
python evaluation/evaluate_uavscenes.py \
  --responses outputs/tcot_uavscenes.jsonl \
  --answer_extractor cot \
  --output outputs/tcot_uavscenes_scores.json
```

## Benchmark Construction

### Open3DVQA metric rank-random


```bash
python scripts/build_open3dvqa_metric_rank_random.py \
  --input "$DATA_ROOT/test_qa.json" \
  --seed  \
  --low-range 0.25 0.70 \
  --high-range 1.40 4.00 \
  --output outputs/open3dvqa_metric_rank_random.json
```
### UAVScenes

```bash
python scripts/build_uavscenes_frame_index.py \
  --data-root /path/to/UAVScenes \
  --calibration-dir /path/to/official/UAVScenes-repository \
  --output artifacts/uavscenes/frame_index.jsonl

python scripts/extract_uavscenes_instances.py \
  --data-root /path/to/UAVScenes \
  --frame-index artifacts/uavscenes/frame_index.jsonl \
  --output artifacts/uavscenes/instances.jsonl \
  --workers 4

python scripts/build_uavscenes_benchmark.py \
  --instances artifacts/uavscenes/instances.jsonl \
  --frame-index artifacts/uavscenes/frame_index.jsonl \
  --seed \
  --output outputs/uavscenes_metric.json
```



