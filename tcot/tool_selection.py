from .answer_prompts import _strip_letter_only_tail
from .tool_schema import O3DVQA_TOOL_MENU_BLOCK

SYSTEM_PROMPT = (
    "You are a spatial-reasoning assistant answering questions about outdoor "
    "urban imagery. This conversation has two turns: (1) request exactly one "
    "external tool result via a <tool_call> block and do NOT answer yet; "
    "(2) answer the original question using the image and any valid external "
    "evidence. Treat tool output as noisy reference evidence, not ground truth."
)


def build_tool_selection_prompt(question):
    menu = O3DVQA_TOOL_MENU_BLOCK
    return (
        f"Question:\n{_strip_letter_only_tail(question or '')}\n\n"
        "First, request one external tool result to help answer the question. "
        "Use a numeric geometry function for metric distance/size/volume "
        "questions. Use relation_evidence for qualitative spatial relation, "
        "direction, left/right/above/below/front/behind, true/false, or "
        "object-choice relation questions; it returns object geometry only, "
        "not the final relation.\n\n" + menu
        + "\n\n### RESPONSE FORMAT (MANDATORY):\n"
        "Emit exactly one <tool_call>...</tool_call> block. Do NOT write any "
        "reasoning, do NOT write A/B/C/D, and do NOT answer the question in this turn."
    )
