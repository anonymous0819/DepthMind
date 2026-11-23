from __future__ import annotations
import re
from typing import Optional

TASK_SIZE = "size"

TASK_CAM_DIST = "cam_dist"

TASK_OBJ_OBJ = "obj_obj"

_O3DVQA_MAP = {
    # single-target size / volume
    "width_data": TASK_SIZE,
    "height_data": TASK_SIZE,
    "volume_data": TASK_SIZE,
    # camera-to-object distance
    "distance2agent": TASK_CAM_DIST,
    "vertical_distance2agent": TASK_CAM_DIST,
    "horizontal_distance2agent": TASK_CAM_DIST,
    # object-to-object distance
    "distance_data": TASK_OBJ_OBJ,
    "vertical_distance_data": TASK_OBJ_OBJ,
    "horizontal_distance_data": TASK_OBJ_OBJ,
}

def map_task_type_o3dvqa(question_name: Optional[str]) -> Optional[str]:
    if not question_name:
        return None
    return _O3DVQA_MAP.get(question_name)

_TMPL_SIZE = (
    "Question:\n__Q__\n\n"
    "Solve this step by step:\n"
    "Step 1. Identify the target object mentioned in the question.\n"
    "Step 2. Recall the typical (canonical) real-world size of that object "
    "in meters. If its size varies a lot (e.g. rooftop, river, field), pick a "
    "nearby reference object in the image whose size is well-known "
    "(e.g. sedan ~4.5 m, lane ~3.5 m wide, person ~1.7 m).\n"
    "Step 3. Compare the target's apparent extent in the image to the anchor "
    "and estimate the target's physical size in meters.\n"
    "Step 4. Match your estimate to one of the four options.\n"
    "Finish with \\boxed{X} (X = A, B, C or D)."
)

_TMPL_CAM_DIST = (
    "Question:\n__Q__\n\n"
    "Solve this step by step:\n"
    "Step 1. Identify the target object the question asks the distance to.\n"
    "Step 2. Recall the typical (canonical) real-world size of that object "
    "in meters. If its size is highly variable, pick a nearby reference object "
    "in the image whose size is well-known (e.g. sedan ~4.5 m, lane ~3.5 m wide, "
    "person ~1.7 m).\n"
    "Step 3. Compare the target's (or the reference's) apparent size on the image "
    "to its real-world size; use this ratio together with UAV viewing geometry to "
    "estimate the distance from the camera to the target in meters.\n"
    "Step 4. Match your estimate to one of the four options.\n"
    "Finish with \\boxed{X} (X = A, B, C or D)."
)

_TMPL_OBJ_OBJ = (
    "Question:\n__Q__\n\n"
    "Solve this step by step:\n"
    "Step 1. Identify the two objects whose distance is being asked.\n"
    "Step 2. Recall the typical (canonical) real-world size of each of them in "
    "meters (or, if the size is highly variable, pick a nearby well-known "
    "reference object such as a sedan ~4.5 m, lane ~3.5 m wide, person ~1.7 m).\n"
    "Step 3. Use those sizes as a ruler: compare the pixel gap between the two "
    "objects to the pixel size of your ruler, then estimate the physical "
    "distance between the two objects in meters.\n"
    "Step 4. Match your estimate to one of the four options.\n"
    "Finish with \\boxed{X} (X = A, B, C or D)."
)

_TMPL_GENERIC = (
    "Question:\n__Q__\n\n"
    "Solve this step by step:\n"
    "Step 1. Identify the object(s) relevant to the question.\n"
    "Step 2. Recall the typical real-world size of each object in meters, and "
    "pick a well-known reference in the image if needed (e.g. sedan ~4.5 m, "
    "lane ~3.5 m wide, person ~1.7 m).\n"
    "Step 3. Use those sizes to reason about the queried quantity in meters.\n"
    "Step 4. Match your estimate to one of the four options.\n"
    "Finish with \\boxed{X} (X = A, B, C or D)."
)

_TASK_TEMPLATES = {
    TASK_SIZE: _TMPL_SIZE,
    TASK_CAM_DIST: _TMPL_CAM_DIST,
    TASK_OBJ_OBJ: _TMPL_OBJ_OBJ,
}

_TAIL_STRIP_PATTERNS = [
    re.compile(
        r"\s*(?:Please\s+)?directly\s+output\s+the\s+single\s+letter[^\n]*\.?\s*"
        r"(?:Do\s+not\s+output\s+anything\s+else\.?)?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*Answer\s+with\s+the\s+letter\s+of\s+the\s+correct\s+option[^\n]*\.?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*Output\s+ONLY\s+the\s+single\s+capital\s+letter[^\n]*\.?\s*$",
        re.IGNORECASE,
    ),
]

def _strip_letter_only_tail(text: str) -> str:
    out = text
    for pat in _TAIL_STRIP_PATTERNS:
        out = pat.sub("", out)
    return out.rstrip()

def build_question_reasoning_prompt(question_text: str, task_type: Optional[str]) -> str:
    """Return the step-by-step user prompt for one MCQ item.

    ``question_text`` is the raw ``mcq_question`` (O3DVQA) or ``question``
    (UAVScenes). It is scrubbed of any tail sentence forcing a single-letter
    reply so the ``\\boxed{X}`` contract is unambiguous.
    """
    q = _strip_letter_only_tail(question_text or "")
    tmpl = _TASK_TEMPLATES.get(task_type or "", _TMPL_GENERIC)
    return tmpl.replace("__Q__", q)

_MCQ_SYSTEM_PROMPT = (
    "You are a spatial-reasoning assistant answering multiple-choice "
    "questions about UAV / aerial imagery. Each question has options "
    "labelled A, B, C or D. When asked, think step by step by recalling "
    "the typical real-world size of familiar objects and using them as "
    "anchors to reason about pixel-to-metric scale."
)

_MCQ_RESPONSE_FORMAT = (
    "\n\n### RESPONSE FORMAT (MANDATORY):\n"
    "You MUST write out ALL FOUR steps above, then finish with "
    "\\boxed{X} on a new line, where X is exactly one of A, B, C or D. "
    "A one-letter answer with no reasoning will be rejected."
)

def build_mcq_reasoning_system_prompt() -> str:
    """Persona-only system prompt used by the Qwen3-VL family."""
    return _MCQ_SYSTEM_PROMPT

def build_mcq_reasoning_prompt(question_text: str, task_type: Optional[str]) -> str:
    """Return the four-step template with the response-format constraint."""
    body = build_question_reasoning_prompt(question_text, task_type)
    return body + _MCQ_RESPONSE_FORMAT
