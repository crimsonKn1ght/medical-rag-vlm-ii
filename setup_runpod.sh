#!/usr/bin/env bash
# One-shot environment setup for the RunPod RTX 6000 Ada box.
# Recommended base image: a CUDA 12.x PyTorch image, Python 3.10/3.11.
set -euo pipefail

PY="${PYTHON:-python}"

echo "==> Python: $($PY --version)"

# 1) (Optional) isolated venv. Comment out if the pod image already has a good env.
if [ "${USE_VENV:-1}" = "1" ]; then
  $PY -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=python
fi

pip install --upgrade pip wheel

# 2) CUDA torch. RTX 6000 Ada works with cu121/cu124 wheels.
#    If the base image already ships torch+cuda, this is a no-op / skip it.
if ! $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  pip install torch --index-url https://download.pytorch.org/whl/cu124 || pip install torch
fi
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 3) Package (editable) + full dependency stack.
pip install -e .
pip install -r requirements-gpu.txt

# 4) Pre-download datasets into the HF cache.
$PY scripts/download_data.py --datasets vqa_rad slake iu_xray || {
  echo "WARN: dataset prefetch failed for one or more datasets; check schemas/access." >&2
}

echo "==> Setup complete. Try:"
echo "    python -m medvlm.eval.run_eval --model llava_med --dataset vqa_rad --split test --n 200"
