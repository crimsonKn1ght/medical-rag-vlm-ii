# medvlm — Year-1 Medical VLM Hallucination Evaluation Harness

Baseline evaluation harness for the research program on **reducing hallucination in
medical vision-language models via retrieval-augmented grounding**. This is the
Year-1 deliverable: run pretrained medical VLMs on radiology VQA + report generation,
and score outputs with automated correctness, factual-consistency, and hallucination
metrics. Years 2–3 (retrieval, comparison grounding, calibration) build on these
interfaces.

## What it does

- **Models** (swappable adapters): `chexagent` (CheXagent-8b), `llava_med`
  (LLaVA-Med v1.5), `smolvlm` (tiny general baseline), `dummy` (offline test).
- **Datasets**: `vqa_rad`, `slake` (VQA), `iu_xray` (report generation, open access).
- **Metrics**:
  - VQA: closed accuracy (EM), open token-F1, semantic similarity.
  - Reports: **CheXbert F1 / RadGraph F1 / GREEN** via [RadEval](https://pypi.org/project/radeval/).
  - Hallucination taxonomy flags: `comparative` (references a prior exam that was
    never given → unsupported by construction), `measurement` (fabricated size),
    `entity` (positive finding vs. negative ground truth).

## Layout

```
configs/            models.yaml, datasets.yaml  (registries)
src/medvlm/
  models/           adapters + registry (uniform generate(image, prompt) -> str)
  data/             loaders -> unified VQAItem / ReportItem
  metrics/          vqa / hallucination / report (RadEval) metrics
  eval/             run_eval (entrypoint), report (aggregation), prompts
  utils/            io, seed, logging
scripts/            download_data.py, smoke_test_cpu.py
results/            run outputs (gitignored)
```

## Quickstart

### Local (CPU) — verify wiring, no downloads
```bash
python scripts/smoke_test_cpu.py            # dummy model + synthetic data, fully offline
```

### RunPod (GPU) — the real runs
```bash
bash setup_runpod.sh                        # torch(cuda) + deps + dataset prefetch

# VQA baselines (two models)
python -m medvlm.eval.run_eval --model llava_med --dataset vqa_rad --split test --n 200
python -m medvlm.eval.run_eval --model chexagent --dataset vqa_rad --split test --n 200

# Report generation + factual metrics (CheXbert / RadGraph / GREEN)
python -m medvlm.eval.run_eval --model chexagent --dataset iu_xray --task report --n 100
```

Each run writes `results/<model>__<dataset>__<ts>/` with `results.jsonl`,
`summary.md`, `examples.md`, and (for report tasks) `report_metrics.yaml`.

## Scope

Delivered: config-driven harness, ≥2 pretrained medical VLMs, VQA + report metrics,
first-cut hallucination taxonomy. **Deferred**: MIMIC-CXR (PhysioNet credentialing),
radiologist-graded validation, RadGraph-entity-level attribution, and all of
Year 2 (retrieval) / Year 3 (comparison grounding + calibrated abstention).

## Notes / gotchas

- **LLaVA-Med**: uses the HF-native `chaoyinshe/llava-med-v1.5-mistral-7b-hf`
  checkpoint (the original `microsoft/...` declares `model_type: llava_mistral`,
  which stock transformers rejects).
- **CheXagent**: custom modeling code (`trust_remote_code=True`); the generation
  format can drift by revision — verify `models/chexagent.py` against the model card
  if generation errors. Requires **transformers < 5** — its remote code imports
  `find_pruneable_heads_and_indices` from `transformers.pytorch_utils`, removed in
  5.x (`requirements-gpu.txt` pins this). If you already installed 5.x, downgrade
  with `pip install "transformers>=4.49,<5"`.
- **RadEval**: confirm the call signature against the installed README; GREEN
  downloads a 7B judge (GPU recommended).
- **SLAKE / IU-Xray**: loaders auto-detect columns across HF mirrors; verify on
  first run.
