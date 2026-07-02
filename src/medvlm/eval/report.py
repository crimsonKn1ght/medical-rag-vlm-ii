"""Aggregate a run's results.jsonl into summary.md + examples.md.

Usable standalone:  python -m medvlm.eval.report results/<run_dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from ..metrics.hallucination import TAXONOMY_FLAGS
from ..utils.io import read_jsonl
from ..utils.logging_utils import get_logger

log = get_logger("medvlm.report")


def _safe_mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if isinstance(v, (int, float))]
    return mean(values) if values else None


def _fmt(x: Optional[float]) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"


def summarize(rows: List[Dict[str, Any]], report_scores: Optional[Dict[str, Any]] = None) -> str:
    n = len(rows)
    task = rows[0].get("task") if rows else "?"
    lines = [f"# Run summary", "", f"- items: **{n}**", f"- task: **{task}**", ""]

    if task == "vqa":
        closed = [r for r in rows if r.get("answer_type") == "closed"]
        openq = [r for r in rows if r.get("answer_type") == "open"]
        lines += [
            "## VQA correctness",
            "",
            f"- closed accuracy (EM): {_fmt(_safe_mean([r.get('correct') for r in closed]))} "
            f"(n={len(closed)})",
            f"- open token-F1: {_fmt(_safe_mean([r.get('correct') for r in openq]))} (n={len(openq)})",
            f"- open semantic similarity: {_fmt(_safe_mean([r.get('semantic_sim') for r in openq]))}",
            f"- overall exact-match: {_fmt(_safe_mean([r.get('exact_match') for r in rows]))}",
            "",
        ]

    # Hallucination taxonomy counts.
    lines += ["## Hallucination taxonomy (automated flags)", ""]
    for flag in TAXONOMY_FLAGS + ["hall_any"]:
        c = sum(1 for r in rows if r.get(flag))
        lines.append(f"- {flag}: {c} / {n} ({100.0 * c / n:.1f}%)" if n else f"- {flag}: 0")
    lines.append("")

    if report_scores:
        lines += ["## Report factual metrics (RadEval)", ""]
        for k, v in _flatten(report_scores).items():
            lines.append(f"- {k}: {_fmt(v) if isinstance(v, (int, float)) else v}")
        lines.append("")

    return "\n".join(lines)


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def examples(rows: List[Dict[str, Any]], k: int = 8) -> str:
    """Qualitative hallucination cases — comparative flags first (cleanest signal)."""
    flagged = [r for r in rows if r.get("hall_any")]
    flagged.sort(key=lambda r: (not r.get("hall_comparative"), not r.get("hall_measurement")))
    lines = ["# Qualitative hallucination examples", ""]
    if not flagged:
        lines.append("_No flagged items._")
        return "\n".join(lines)
    for r in flagged[:k]:
        fired = [f for f in TAXONOMY_FLAGS if r.get(f)]
        lines += [
            f"## item {r.get('id')}  ({', '.join(fired)})",
            f"- Q: {r.get('question', '(report)')}",
            f"- gold: {r.get('answer', r.get('reference', ''))[:200]}",
            f"- pred: {r.get('prediction', '')[:300]}",
            "",
        ]
    return "\n".join(lines)


def aggregate_run(run_dir: str | Path, report_scores: Optional[Dict[str, Any]] = None) -> None:
    run_dir = Path(run_dir)
    rows = read_jsonl(run_dir / "results.jsonl")
    if not rows:
        log.warning("no rows in %s", run_dir)
        return
    (run_dir / "summary.md").write_text(summarize(rows, report_scores), encoding="utf-8")
    (run_dir / "examples.md").write_text(examples(rows), encoding="utf-8")
    log.info("wrote summary.md + examples.md to %s", run_dir)


def main():
    p = argparse.ArgumentParser(description="Aggregate a results.jsonl run directory")
    p.add_argument("run_dir")
    args = p.parse_args()
    aggregate_run(args.run_dir)


if __name__ == "__main__":
    main()
