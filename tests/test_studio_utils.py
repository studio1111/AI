import pytest

from studio_utils import (
    ASPECT_RATIOS,
    DEFAULT_NEGATIVE_PROMPT,
    build_image_prompts,
    clean_text,
    resolve_dimensions,
)


def test_clean_text_trims_strings_and_handles_none():
    assert clean_text("  hello  ") == "hello"
    assert clean_text(None) == ""


def test_clean_text_supports_gradio_message_shapes():
    assert clean_text({"content": "  hi "}) == "hi"
    assert clean_text(["hello", {"text": "world"}]) == "hello world"
    assert clean_text((" user ", " assistant ")) == "user assistant"


def test_image_prompt_requires_nonempty_text():
    with pytest.raises(ValueError, match="image prompt"):
        build_image_prompts("  ")


def test_image_prompt_includes_default_and_custom_negative_terms():
    positive, negative = build_image_prompts("  watercolor fox ", " extra limbs ")
    assert positive == "watercolor fox"
    assert DEFAULT_NEGATIVE_PROMPT in negative
    assert negative.endswith("extra limbs")


def test_all_image_dimensions_are_valid_and_resolvable():
    assert set(ASPECT_RATIOS) == {
        "16:9 Landscape", "1:1 Square", "9:16 Portrait", "4:3 Classic"
    }
    for label, dimensions in ASPECT_RATIOS.items():
        assert resolve_dimensions(label) == dimensions
        assert all(value % 8 == 0 for value in dimensions)


def test_unknown_image_ratio_fails_clearly():
    with pytest.raises(ValueError, match="Unsupported image ratio"):
        resolve_dimensions("panorama")
