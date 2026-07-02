"""Dummy adapter for offline pipeline testing — no weights, no download.

It produces deterministic text derived from the prompt, and deliberately emits a
comparative phrase on a fraction of items so the hallucination flags have
something to fire on during the smoke test.
"""
from __future__ import annotations

from PIL.Image import Image

from .base import VLMAdapter


class DummyAdapter(VLMAdapter):
    name = "dummy"

    def load(self) -> "DummyAdapter":
        self.model = "dummy"
        return self

    def generate(self, image: Image, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        # Cheap deterministic behaviour keyed off the prompt length.
        h = len(prompt)
        if h % 3 == 0:
            return "no"
        if h % 3 == 1:
            return "yes"
        # Inject an unsupported comparative claim + a fabricated measurement.
        return "There is a 2.5 cm nodule, unchanged compared to the prior study."
