"""First-cut automated hallucination flags -> the Year-1 taxonomy seed.

Three categories from the research plan:

- comparative: the model references a prior/previous exam, interval change, or
  temporal progression. For single-image VQA/report input, no prior was provided,
  so any such statement is UNSUPPORTED by construction. This is a clean, precise
  automated hallucination detector.
- measurement: the model states a numeric size/measurement (mm, cm, %) that does
  not appear in the question — a likely fabricated quantity.
- entity: coarse placeholder — the model asserts a positive finding while the
  ground-truth answer is negative/normal. Upgraded to RadGraph-entity overlap in a
  later Year-1 step.

Each flag is a boolean; aggregation elsewhere turns these into taxonomy counts.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

TAXONOMY_FLAGS = ["hall_comparative", "hall_measurement", "hall_entity"]

_COMPARATIVE_TERMS = [
    "prior", "previous", "compared", "comparison", "interval", "unchanged",
    "again", "worsened", "worsening", "improved", "improvement", "since",
    "follow-up", "followup", "stable", "increased", "decreased", "no longer",
    "re-demonstrated", "redemonstrated", "persistent",
]
_COMPARATIVE_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in _COMPARATIVE_TERMS) + r")\b", re.I)

_MEASUREMENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mm|cm|millimeter|millimetre|centimeter|centimetre|%)\b", re.I)

_NEGATION_RE = re.compile(r"\b(no|not|without|negative|normal|clear|unremarkable|absent)\b", re.I)
# A few common radiology finding words for the coarse entity check.
_FINDING_TERMS = [
    "pneumothorax", "effusion", "consolidation", "edema", "oedema", "opacity",
    "nodule", "mass", "fracture", "cardiomegaly", "atelectasis", "infiltrate",
    "pneumonia", "emphysema", "lesion", "hemorrhage", "haemorrhage",
]
_FINDING_RE = re.compile(r"\b(" + "|".join(_FINDING_TERMS) + r")\b", re.I)


def flag_comparative(pred: str, has_prior: bool = False) -> bool:
    if has_prior:
        return False
    return bool(_COMPARATIVE_RE.search(pred or ""))


def flag_measurement(pred: str, question: str = "") -> bool:
    pred_hits = set(m.group(0).lower().replace(" ", "") for m in _MEASUREMENT_RE.finditer(pred or ""))
    if not pred_hits:
        return False
    q_hits = set(m.group(0).lower().replace(" ", "") for m in _MEASUREMENT_RE.finditer(question or ""))
    return bool(pred_hits - q_hits)


def flag_entity(pred: str, gold: Optional[str] = None) -> bool:
    """Positive finding asserted where the ground truth is negative/normal."""
    if not _FINDING_RE.search(pred or ""):
        return False
    if _NEGATION_RE.search(pred or ""):
        return False  # model itself negated it
    if gold is None:
        return False
    gold_negative = bool(_NEGATION_RE.search(gold)) or str(gold).strip().lower() in {"no", "none", "normal"}
    return gold_negative


def score_hallucination(pred: str, item, has_prior: bool = False) -> Dict[str, Any]:
    question = getattr(item, "question", "") or ""
    gold = getattr(item, "answer", None)
    flags = {
        "hall_comparative": flag_comparative(pred, has_prior=has_prior),
        "hall_measurement": flag_measurement(pred, question),
        "hall_entity": flag_entity(pred, gold),
    }
    flags["hall_any"] = any(flags[k] for k in TAXONOMY_FLAGS)
    return flags
