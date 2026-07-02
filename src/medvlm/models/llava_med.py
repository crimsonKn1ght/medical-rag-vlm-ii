"""LLaVA-Med adapter (HF-native conversion).

Uses `chaoyinshe/llava-med-v1.5-mistral-7b-hf`, which loads with
`LlavaForConditionalGeneration`. The original `microsoft/llava-med-v1.5-mistral-7b`
declares `model_type: llava_mistral`, which stock transformers rejects — hence the
converted checkpoint here.
"""
from __future__ import annotations

from PIL.Image import Image

from .base import VLMAdapter, resolve_dtype


class LlavaMedAdapter(VLMAdapter):
    name = "llava_med"

    def load(self) -> "LlavaMedAdapter":
        import torch  # noqa: F401
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        torch_dtype = resolve_dtype(self.dtype_str, self.device)
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.hf_id, torch_dtype=torch_dtype
        ).to(self.device)
        self.model.eval()
        return self

    def generate(self, image: Image, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        import torch

        self._ensure_loaded()
        # LLaVA-1.5 mistral conversation format.
        text = f"USER: <image>\n{prompt} ASSISTANT:"
        inputs = self.processor(images=image.convert("RGB"), text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        decoded = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        # Return only the assistant turn.
        if "ASSISTANT:" in decoded:
            decoded = decoded.split("ASSISTANT:", 1)[1]
        return decoded.strip()
