"""SmolVLM adapter — tiny, CPU-friendly general VLM.

Used for the local smoke test and as an optional general-domain baseline on the
pod (general-vs-specialist comparison the plan mentions).
"""
from __future__ import annotations

from PIL.Image import Image

from .base import VLMAdapter, resolve_dtype


class SmolVLMAdapter(VLMAdapter):
    name = "smolvlm"

    def load(self) -> "SmolVLMAdapter":
        import torch  # noqa: F401
        from transformers import AutoModelForVision2Seq, AutoProcessor

        torch_dtype = resolve_dtype(self.dtype_str, self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.hf_id, trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.hf_id,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
        ).to(self.device)
        self.model.eval()
        return self

    def generate(self, image: Image, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        import torch

        self._ensure_loaded()
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image.convert("RGB")], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
