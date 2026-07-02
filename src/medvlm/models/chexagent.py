"""CheXagent-8b adapter (Stanford chest-Xray foundation model).

CheXagent ships custom modeling code (`trust_remote_code=True`) and a custom
prompt format built via the processor's tokenizer. The generation path below
follows the published model-card usage. NOTE: this API is custom and can drift
between checkpoint revisions — verify against the current model card on the pod
if generation errors, and adjust `_build_prompt` accordingly.
"""
from __future__ import annotations

from PIL.Image import Image

from .base import VLMAdapter, resolve_dtype

_SYSTEM = (
    "You are a helpful assistant. You are able to understand the chest X-ray "
    "image the user provides and assist the user with radiology tasks."
)


class CheXagentAdapter(VLMAdapter):
    name = "chexagent"

    def load(self) -> "CheXagentAdapter":
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        torch_dtype = resolve_dtype(self.dtype_str, self.device)
        self.processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.hf_id)
        except Exception:
            self.generation_config = None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # CheXagent's generation config leaves pad_token_id unset, so transformers
        # logs "Setting `pad_token_id` to `eos_token_id`" on every generate() call.
        # Pin it once to eos to keep long-run logs readable.
        eos_id = self.processor.tokenizer.eos_token_id
        for cfg in (self.generation_config, getattr(self.model, "generation_config", None)):
            if cfg is not None and cfg.pad_token_id is None:
                cfg.pad_token_id = eos_id
        return self

    def _build_inputs(self, image: Image, prompt: str):
        # CheXagent's processor expects images + a text query. The card uses
        # `processor(images=..., text=..., return_tensors="pt")`.
        return self.processor(
            images=[image.convert("RGB")], text=f" USER: <s>{prompt} ASSISTANT: <s>",
            return_tensors="pt",
        )

    def generate(self, image: Image, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        import torch

        self._ensure_loaded()
        inputs = self._build_inputs(image, prompt)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )[0]
        gen = out[input_len:] if input_len else out
        text = self.processor.tokenizer.decode(gen, skip_special_tokens=True)
        return text.strip()
