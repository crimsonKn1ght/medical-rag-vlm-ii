"""Model name -> adapter factory."""
from __future__ import annotations

from typing import Any, Dict

from .base import VLMAdapter
from .chexagent import CheXagentAdapter
from .dummy import DummyAdapter
from .llava_med import LlavaMedAdapter
from .smolvlm import SmolVLMAdapter

ADAPTERS = {
    "dummy": DummyAdapter,
    "smolvlm": SmolVLMAdapter,
    "llava_med": LlavaMedAdapter,
    "chexagent": CheXagentAdapter,
}


def build_adapter(model_name: str, models_cfg: Dict[str, Any], device: str | None = None) -> VLMAdapter:
    if model_name not in models_cfg:
        raise KeyError(f"Model '{model_name}' not in models.yaml. Known: {list(models_cfg)}")
    cfg = dict(models_cfg[model_name])
    adapter_key = cfg.pop("adapter")
    if adapter_key not in ADAPTERS:
        raise KeyError(f"Adapter '{adapter_key}' unknown. Known: {list(ADAPTERS)}")
    hf_id = cfg.pop("hf_id", "")
    return ADAPTERS[adapter_key](hf_id=hf_id, device=device, **cfg)
