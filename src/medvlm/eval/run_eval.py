"""Main evaluation entrypoint.

    python -m medvlm.eval.run_eval --model llava_med --dataset vqa_rad --split test --n 200
    python -m medvlm.eval.run_eval --model chexagent --dataset iu_xray --task report --n 100
    python -m medvlm.eval.run_eval --model dummy --dataset synthetic          # offline test

Produces results/<model>__<dataset>__<ts>/{results.jsonl, config.yaml, summary.md, examples.md}.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..data.loaders import load_dataset_items, load_synthetic_vqa
from ..metrics.hallucination import score_hallucination
from ..metrics.vqa_metrics import score_vqa
from ..models.registry import build_adapter
from ..utils.io import JsonlWriter, ensure_dir, read_yaml, timestamp
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from .prompts import report_prompt, vqa_prompt
from .report import aggregate_run

log = get_logger("medvlm.eval")

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_OUT = REPO_ROOT / "results"


def _load_items(args, datasets_cfg):
    if args.dataset == "synthetic":
        return "vqa", load_synthetic_vqa(args.n or 6)
    if args.dataset not in datasets_cfg:
        raise KeyError(f"Dataset '{args.dataset}' not in datasets.yaml: {list(datasets_cfg)}")
    ds_cfg = dict(datasets_cfg[args.dataset])
    if args.task:
        ds_cfg["task"] = args.task
    return load_dataset_items(ds_cfg, args.split, args.n)


def run(args) -> Path:
    set_seed(args.seed)
    models_cfg = read_yaml(args.models_config)
    datasets_cfg = read_yaml(args.datasets_config)

    task, items = _load_items(args, datasets_cfg)
    log.info("Task=%s | %d items | model=%s", task, len(items), args.model)

    adapter = build_adapter(args.model, models_cfg, device=args.device)
    if args.model != "dummy":
        log.info("Loading model %s ...", args.model)
    adapter.load()

    out_dir = ensure_dir(Path(args.out) / f"{args.model}__{args.dataset}__{timestamp()}")
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump({**vars(args), "task": task, "n_items": len(items)}), encoding="utf-8"
    )

    writer = JsonlWriter(out_dir / "results.jsonl")
    report_refs: List[str] = []
    report_hyps: List[str] = []
    t0 = time.time()

    for idx, item in enumerate(items):
        if task == "vqa":
            prompt = vqa_prompt(item.question, item.answer_type)
        else:
            prompt = report_prompt()

        try:
            pred = adapter.generate(item.image, prompt, max_new_tokens=args.max_new_tokens)
        except Exception as e:  # keep long runs alive; record the failure
            log.warning("generate failed on item %s: %s", item.id, e)
            pred = ""

        row: Dict[str, Any] = {"id": item.id, "task": task, "prompt": prompt, "prediction": pred}

        if task == "vqa":
            row["question"] = item.question
            row["answer"] = item.answer
            row.update(score_vqa(pred, item, use_semantic=not args.no_semantic))
        else:
            row["reference"] = item.reference
            report_refs.append(item.reference)
            report_hyps.append(pred)

        # Hallucination flags apply to both tasks (single-image input => no prior).
        row.update(score_hallucination(pred, item if task == "vqa" else _ReportShim(pred)))
        writer.write(row)

        if (idx + 1) % 25 == 0:
            log.info("  %d/%d done (%.1fs)", idx + 1, len(items), time.time() - t0)

    writer.close()

    # Report-level factual metrics (CheXbert / RadGraph / GREEN) via RadEval.
    report_scores = None
    if task == "report" and not args.no_report_metrics and report_hyps:
        try:
            from ..metrics.report_metrics import compute_report_metrics

            log.info("Computing RadEval report metrics on %d pairs ...", len(report_hyps))
            report_scores = compute_report_metrics(report_refs, report_hyps)
            (out_dir / "report_metrics.yaml").write_text(
                yaml.safe_dump(_to_plain(report_scores)), encoding="utf-8"
            )
        except Exception as e:
            log.warning("RadEval metrics skipped: %s", e)

    aggregate_run(out_dir, report_scores=report_scores)
    log.info("Done in %.1fs -> %s", time.time() - t0, out_dir)
    return out_dir


class _ReportShim:
    """Adapts a report prediction to the hallucination scorer's item interface."""

    def __init__(self, pred: str):
        self.question = ""
        self.answer = None


def _to_plain(obj):
    """Make RadEval outputs YAML-serialisable."""
    try:
        import numpy as np

        if isinstance(obj, dict):
            return {k: _to_plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_plain(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
    except Exception:
        pass
    return obj


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Medical VLM baseline evaluation")
    p.add_argument("--model", required=True, help="key in configs/models.yaml (or 'dummy')")
    p.add_argument("--dataset", required=True, help="key in configs/datasets.yaml (or 'synthetic')")
    p.add_argument("--task", default=None, choices=[None, "vqa", "report"], help="override dataset task")
    p.add_argument("--split", default=None)
    p.add_argument("--n", type=int, default=None, help="max items (None = all)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default=None, help="cuda|cpu (auto by default)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-semantic", action="store_true", help="skip sentence-transformer similarity")
    p.add_argument("--no-report-metrics", action="store_true", help="skip RadEval (CheXbert/RadGraph/GREEN)")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="output root (default: <repo>/results, regardless of cwd)")
    p.add_argument("--models-config", default=str(CONFIG_DIR / "models.yaml"))
    p.add_argument("--datasets-config", default=str(CONFIG_DIR / "datasets.yaml"))
    return p


def main():
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
