"""Radiology report factual-consistency metrics via RadEval.

RadEval bundles F1CheXbert, F1RadGraph, and GREEN. The clinical models download
from HF on first use (GREEN is a 7B judge — wants GPU). Import and construction
are lazy so the rest of the harness runs without RadEval installed.

The exact RadEval call signature can vary by version; this wrapper tries the
documented `RadEval(metrics=...)(refs=..., hyps=...)` path and surfaces a clear
error otherwise so it can be adjusted against the installed README.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..utils.logging_utils import get_logger

log = get_logger("medvlm.metrics.report")

_EVALUATOR = None


def _get_evaluator(metrics: List[str]):
    global _EVALUATOR
    if _EVALUATOR is not None:
        return _EVALUATOR
    from radeval import RadEval  # noqa: F401

    _EVALUATOR = RadEval(metrics=metrics)
    return _EVALUATOR


def compute_report_metrics(
    refs: List[str],
    hyps: List[str],
    metrics: Optional[List[str]] = None,
    per_sample: bool = False,
) -> Dict[str, Any]:
    """Score generated reports against references.

    Returns a dict of metric_name -> score (aggregate) and, if available and
    requested, per-sample values under 'per_sample'.
    """
    metrics = metrics or ["f1chexbert", "radgraph", "green"]
    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) and hyps ({len(hyps)}) length mismatch")
    try:
        evaluator = _get_evaluator(metrics)
    except ImportError as e:
        raise ImportError(
            "radeval is not installed. `pip install radeval` (see requirements-gpu.txt)."
        ) from e

    # Documented signature: evaluator(refs=..., hyps=...). Fall back to positional.
    try:
        return evaluator(refs=refs, hyps=hyps)
    except TypeError:
        log.warning("RadEval keyword call failed; retrying positional. Verify installed API.")
        return evaluator(refs, hyps)
