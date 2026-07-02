"""Pre-download datasets into the HF cache so the first eval run doesn't stall.

    python scripts/download_data.py --datasets vqa_rad slake iu_xray
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medvlm.utils.io import read_yaml  # noqa: E402
from medvlm.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("medvlm.download")
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "datasets.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["vqa_rad", "slake", "iu_xray"])
    p.add_argument("--config", default=str(CONFIG))
    args = p.parse_args()

    cfg = read_yaml(args.config)
    from datasets import load_dataset

    for name in args.datasets:
        if name not in cfg:
            log.warning("skip unknown dataset '%s'", name)
            continue
        hf_id = cfg[name]["hf_id"]
        log.info("downloading %s (%s) ...", name, hf_id)
        try:
            ds = load_dataset(hf_id)
            splits = {k: len(v) for k, v in ds.items()}
            log.info("  OK %s -> splits=%s cols=%s", name, splits, ds[list(ds)[0]].column_names)
        except Exception as e:
            log.warning("  FAILED %s: %s", name, e)


if __name__ == "__main__":
    main()
