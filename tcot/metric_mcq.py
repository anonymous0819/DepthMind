from __future__ import annotations

from .answer_prompts import extract_letter_cot
from .question_prompts import map_task_type_o3dvqa
from .o3dvqa_protocol import build_mcq_no_tool_prompt, build_mcq_system_prompt, build_metric_tool_fusion_prompt
from .qwen3vl_runtime import append_tool_history, generate_text, make_messages


def build_metric_prompt(question, question_name, tool_result=None):
    task = map_task_type_o3dvqa(question_name)
    if tool_result is None:
        return build_mcq_no_tool_prompt(question, task)
    return build_metric_tool_fusion_prompt(question, task, tool_result)


def run(bundle, image, question, question_name, depth_path, tool_result=None, tool_messages=None, tool_pick=None):
    usable = tool_result is not None and tool_result.status == "ok" and tool_result.value is not None
    system = build_mcq_system_prompt()
    prompt = build_metric_prompt(question, question_name, tool_result if usable else None)
    if usable and tool_messages is not None and tool_pick is not None:
        messages = append_tool_history(tool_messages, tool_pick, prompt)
    else:
        messages = make_messages(image, prompt, system)
    raw = generate_text(bundle, messages, depth_path)
    return dict(response=raw, pred_letter=extract_letter_cot(raw),
                branch="metric_tool_ok" if usable else "cot_fallback")
