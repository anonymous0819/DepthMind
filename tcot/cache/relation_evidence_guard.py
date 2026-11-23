from __future__ import annotations

import re

FAMILIES = {"left", "right", "above", "below", "front", "behind", "big", "small", "wide", "thin", "tall", "short"}


def question_context(question, question_name, evidence):
    stem = question.split("\n", 1)[0]
    objects = evidence.get("objects") or {}
    for obj in sorted((obj for obj in objects.values() if isinstance(obj, dict)),
                      key=lambda obj: len(obj.get("prompt", "")), reverse=True):
        phrase = re.sub(r"^\s*(?:the|a|an)\s+", "", obj.get("prompt", ""), flags=re.I)
        if phrase:
            stem = re.sub(r"(?<!\w)(?:the\s+|a\s+|an\s+)?" + re.escape(phrase) + r"(?!\w)", " OBJECT ", stem, flags=re.I)
    qname = question_name or ""
    if qname in {"direction_data", "direction2agent"} or re.search(r"\bwhere will you find\b|\bo['’]?clock\b", stem, re.I):
        family = "clock"
    else:
        family = qname.split("_", 1)[0]
        if family not in FAMILIES:
            match = re.search(r"\b(above|below|under|over|higher|lower|left|right|front|behind|nearer|farther|further|wider|thinner|taller|shorter|bigger|smaller)\b", stem, re.I)
            aliases = {"under": "below", "over": "above", "higher": "above", "lower": "below", "nearer": "front",
                       "farther": "behind", "further": "behind", "wider": "wide", "thinner": "thin", "taller": "tall",
                       "shorter": "short", "bigger": "big", "smaller": "small"}
            family = aliases.get(match[1].lower(), match[1].lower()) if match else "unknown"
    screen = bool(re.search(r"\b(?:image|screen|photo|picture|pixel)(?:[- ](?:relative|plane|coordinates?))?\b", stem, re.I))
    viewer = bool(re.search(r"\b(?:viewer|camera)\b", stem, re.I))
    facing = bool(re.search(r"\b(?:facing|orientation|heading|world|gravity|physical|physically|altitude|north|south|east|west)\b", stem, re.I))
    agent = "2agent" in qname or bool(re.search(r"\b(?:of you|of me|your (?:left|right)|my (?:left|right))\b", stem, re.I))
    return dict(family=family, explicit_image_frame=screen, explicit_viewer_frame=viewer,
                orientation_frame=facing, agent_reference=agent)
