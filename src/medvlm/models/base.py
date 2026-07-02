"""Uniform VLM adapter interface.

Every model exposes the same tiny surface so the eval loop and (later) the
Year-2 retrieval layer can wrap any model identically:

    adapter = SomeAdapter(hf_id=..., dtype=...)
    adapter.load()
    text = adapter.generate(image, prompt)
"""
from __future__ import annotations

from typing import Optional

from PIL.Image import Image


def resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_dtype(dtype: str, device: str):
    """Map a dtype string to a torch dtype, forcing float32 on CPU."""
    try:
        import torch
    except Exception:
        return None
    if device == "cpu":
        return torch.float32
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(dtype).lower(), torch.float32)


class VLMAdapter:
    name: str = "base"

    def __init__(
        self,
        hf_id: str,
        dtype: str = "float32",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        self.hf_id = hf_id
        self.dtype_str = dtype
        self.device = resolve_device(device)
        self.trust_remote_code = trust_remote_code
        self.kwargs = kwargs
        self.model = None
        self.processor = None

    def load(self) -> "VLMAdapter":
        raise NotImplementedError

    def generate(self, image: Image, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        raise NotImplementedError

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load()
