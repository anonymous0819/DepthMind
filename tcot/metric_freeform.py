from __future__ import annotations

from drope_common import SYSTEM_PROMPT as FREEFORM_SYSTEM_PROMPT
from .qwen3vl_runtime import append_tool_history, generate_text, make_messages
from .o3dvqa_protocol import build_universal_tool_first_system_prompt


def build_metric_prompt(question, tool_result=None):
    if tool_result is None:
        prompt = (
            "Answer the question using only the image. Keep the final answer "
            "concise and in the same style as the question expects.\n\n"
            f"Question: {question}"
        )
    else:
        prompt = (
            "A numeric geometry tool has been called for you and returned:\n"
            f"<tool_response>{tool_result.to_prompt_json()}</tool_response>\n"
            "If a `quantity` field is present, it states exactly what was measured. "
            "The measurement may be noisy or wrong; use it as a soft reference "
            "together with the image. Answer with the requested free-form value "
            "or phrase, without exposing internal tool JSON.\n\n"
            f"Question: {question}"
        )
    return prompt


def run(bundle, image, question, depth_path, tool_result=None, tool_messages=None, tool_pick=None):
    usable = tool_result is not None and tool_result.status == "ok" and tool_result.value is not None
    prompt = build_metric_prompt(question, tool_result if usable else None)
    if usable and tool_messages is not None and tool_pick is not None:
        messages = append_tool_history(tool_messages, tool_pick, prompt)
    else:
        system = build_universal_tool_first_system_prompt() if usable else FREEFORM_SYSTEM_PROMPT
        messages = make_messages(image, prompt, system)
    raw = generate_text(bundle, messages, depth_path)
    return dict(response=raw, branch="metric_tool_ok" if usable else "cot_fallback")
