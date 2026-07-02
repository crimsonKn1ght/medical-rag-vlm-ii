"""Offline end-to-end pipeline test — no network, no GPU, no model download.

Runs the full eval path (synthetic items -> dummy adapter -> VQA metrics +
hallucination flags -> results.jsonl -> summary/examples) and asserts the wiring
holds. This is the local gate before pushing to the pod.

    python scripts/smoke_test_cpu.py
    python scripts/smoke_test_cpu.py --model smolvlm   # also exercise a real tiny VLM (downloads ~1GB)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medvlm.eval.run_eval import build_argparser, run  # noqa: E402
from medvlm.utils.io import read_jsonl  # noqa: E402
from medvlm.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("medvlm.smoke")

REQUIRED_VQA_FIELDS = ["prediction", "exact_match", "token_f1", "correct",
                       "hall_comparative", "hall_measurement", "hall_entity", "hall_any"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dummy", help="dummy (offline) or smolvlm (downloads)")
    p.add_argument("--n", type=int, default=6)
    cli = p.parse_args()

    # Build run args via the eval argparser so we exercise the real code path.
    argv = [
        "--model", cli.model,
        "--dataset", "synthetic",
        "--n", str(cli.n),
        "--no-report-metrics",
        "--no-semantic",  # keep fully offline for the dummy path
        "--out", "results/smoke",
    ]
    args = build_argparser().parse_args(argv)

    out_dir = run(args)
    rows = read_jsonl(out_dir / "results.jsonl")

    assert rows, "no result rows written"
    assert len(rows) == cli.n, f"expected {cli.n} rows, got {len(rows)}"
    for r in rows:
        for f in REQUIRED_VQA_FIELDS:
            assert f in r, f"missing field '{f}' in row {r.get('id')}"
    assert (out_dir / "summary.md").exists(), "summary.md not written"
    assert (out_dir / "examples.md").exists(), "examples.md not written"

    n_flagged = sum(1 for r in rows if r["hall_any"])
    log.info("PASS: %d rows, %d hallucination-flagged, artifacts in %s", len(rows), n_flagged, out_dir)
    print("\nSMOKE TEST PASSED ->", out_dir)


if __name__ == "__main__":
    main()
