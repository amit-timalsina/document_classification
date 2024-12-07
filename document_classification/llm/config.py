from __future__ import annotations

from pathlib import Path

# Configuration

OCR_JSON_DIRECTORY = Path("ocr_jsons_tesseract")


class LLMConfig:
    """Configuration for the LLM package."""

    model_name: str
    temperature: float
    max_tokens: int
    logprobs: bool
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    top_logprobs: int
    json_mode: bool
