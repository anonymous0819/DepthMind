"""Reasoning prompts and final-option extraction for Open3DVQA."""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Task-type mapping
# ---------------------------------------------------------------------------

TASK_SIZE = "size"          # width / height / horizontal extent of a single target
TASK_VOLUME = "volume"
TASK_CAM_DIST = "cam_dist"  # distance from camera / observer to a single target
TASK_OBJ_OBJ = "obj_obj"    # distance between two objects

_O3DVQA_MAP = {
    # single-target size / volume
    "width_data": TASK_SIZE,
    "height_data": TASK_SIZE,
    "volume_data": TASK_VOLUME,
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


# ---------------------------------------------------------------------------
# System prompt (persona-only)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a spatial-reasoning assistant answering multiple-choice "
    "questions about UAV / aerial imagery. Each question has options "
    "labelled A, B, C or D. When asked, think step by step by recalling "
    "the typical real-world size of familiar objects and using them as "
    "anchors to reason about pixel-to-metric scale."
)


# ---------------------------------------------------------------------------
# Per-task CoT templates (body)
# ---------------------------------------------------------------------------
# We do NOT strip the original MCQ question text; we prepend an instruction
# so the model still sees the full "Options A/B/C/D" listing verbatim.

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

_TMPL_VOLUME = (
    "Question:\n__Q__\n\n"
    "Solve this step by step:\n"
    "Step 1. Identify the target object and the volumetric extent asked for.\n"
    "Step 2. Recall plausible real-world dimensions in meters, using a familiar "
    "nearby object as a scale anchor when necessary.\n"
    "Step 3. Estimate the target's width, height and depth, then estimate its "
    "volume in cubic meters using the appropriate shape. Do not substitute "
    "a single length in meters for volume.\n"
    "Step 4. Match the volume estimate to one of the four volume options.\n"
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

# Generic fallback for items whose task type is unknown / unmapped.
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
    TASK_VOLUME: _TMPL_VOLUME,
    TASK_CAM_DIST: _TMPL_CAM_DIST,
    TASK_OBJ_OBJ: _TMPL_OBJ_OBJ,
}


# ---------------------------------------------------------------------------
# Hard response-format tail (user-turn only; see module docstring).
# ---------------------------------------------------------------------------

_HARD_TAIL = (
    "\n\n### RESPONSE FORMAT (MANDATORY):\n"
    "You MUST write out ALL FOUR steps above, then finish with "
    "\\boxed{X} on a new line, where X is exactly one of A, B, C or D. "
    "A one-letter answer with no reasoning will be rejected."
)


# ---------------------------------------------------------------------------
# Tail-strip for datasets that append a "single-letter only" instruction
# to their MCQ question text (incompatible with the \boxed{X} contract).
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_answer_reasoning_system_prompt() -> str:
    """Persona-only system prompt."""
    return _SYSTEM_PROMPT


def build_answer_reasoning_prompt(
    question_text: str, task_type: Optional[str],
) -> str:
    """Step-by-step user prompt for one MCQ item.

    The raw ``mcq_question`` is scrubbed of any tail sentence that forces a single-letter
    reply, then wrapped with the per-task four-step body, then suffixed
    with the MANDATORY response-format block.
    """
    q = _strip_letter_only_tail(question_text or "")
    tmpl = _TASK_TEMPLATES.get(task_type or "", _TMPL_GENERIC)
    return tmpl.replace("__Q__", q) + _HARD_TAIL


# ---------------------------------------------------------------------------
# Letter extractor
# ---------------------------------------------------------------------------

# Priority 1: \boxed{X} (with optional whitespace / trailing punctuation).
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([ABCD])\s*[.)\]]?\s*\}", re.IGNORECASE)
# Priority 2: "final answer(:)? X" / "answer is X" / "the answer is X" etc.
_FINAL_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)?\s*[\"'`\(\[]?\s*([ABCD])\b",
    re.IGNORECASE,
)
# Priority 3: last standalone A/B/C/D anywhere in the string.
_ANY_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def extract_letter_cot(response: str) -> Optional[str]:
    """Robustly extract A/B/C/D from a CoT (or plain) response.

    Priority order:
        1. ``\\boxed{X}`` (case-insensitive, tolerant of trailing punct).
        2. ``final answer / answer is / answer: X``.
        3. Last standalone A/B/C/D anywhere in the response.
        4. None (abstain).

    Note: for pure-letter responses like ``"A"`` or ``"A."`` the fallback
    (3) already recovers the letter, so the plain-MCQ baselines are also
    correctly parsed by this function.
    """
    if not response:
        return None
    m = _BOXED_RE.search(response)
    if m:
        return m.group(1).upper()
    m = _FINAL_RE.search(response)
    if m:
        return m.group(1).upper()
    # Take the LAST standalone letter, not the first — for CoT the answer is
    # at the end, while intermediate mentions like "option A costs 4.5 m" are
    # early in the text.
    matches = _ANY_LETTER_RE.findall(response)
    if matches:
        return matches[-1].upper()
    return None


