"""Task-specific prompt templates.

Kept intentionally simple/model-agnostic for the baseline; per-model tuning can be
added later without touching the eval loop.
"""
from __future__ import annotations


def vqa_prompt(question: str, answer_type: str = "open") -> str:
    if answer_type == "closed":
        return (
            f"{question}\nAnswer with 'yes' or 'no' only, based only on the image."
        )
    return (
        f"{question}\nAnswer concisely based only on the image. "
        f"If the image does not show it, say so."
    )


def report_prompt() -> str:
    return (
        "Describe the findings in this chest X-ray and give an impression. "
        "Report only what is visible in this single image."
    )
