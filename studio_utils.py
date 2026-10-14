"""Small, dependency-light helpers shared by AI Studio and its tests."""
from __future__ import annotations

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "16:9 Landscape": (1216, 684),
    "1:1 Square": (1024, 1024),
    "9:16 Portrait": (684, 1216),
    "4:3 Classic": (1152, 864),
}

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, jpeg artifacts, malformed hands, extra fingers, "
    "missing fingers, distorted anatomy, text, watermark"
)


def clean_text(value: object) -> str:
    """Normalize Gradio chat values and legacy tuple-based history."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return clean_text(value.get("text", value.get("content", "")))
    if isinstance(value, (list, tuple)):
        return " ".join(part for part in (clean_text(item) for item in value) if part).strip()
    return str(value).strip()


def build_image_prompts(prompt: str, negative_prompt: str = "") -> tuple[str, str]:
    """Add quality guidance while leaving the user's prompt intact."""
    positive = clean_text(prompt)
    if not positive:
        raise ValueError("Please enter an image prompt.")
    negative = DEFAULT_NEGATIVE_PROMPT
    extra = clean_text(negative_prompt)
    if extra:
        negative = f"{negative}, {extra}"
    return positive, negative


def resolve_dimensions(label: str) -> tuple[int, int]:
    try:
        return ASPECT_RATIOS[label]
    except KeyError as exc:
        raise ValueError(f"Unsupported image ratio: {label}") from exc
