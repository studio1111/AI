"""AI Studio: Google Colab-friendly text chat and image generation UI."""
from __future__ import annotations

import gc
import os
import threading
from typing import Any

import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from studio_utils import (
    ASPECT_RATIOS,
    build_image_prompts,
    clean_text,
    resolve_dimensions,
)

CHAT_MODEL_ID = os.getenv("CHAT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
IMAGE_STEPS_DEFAULT = int(os.getenv("IMAGE_STEPS_DEFAULT", "25"))

_chat_model: Any = None
_chat_tokenizer: Any = None
_image_pipe: Any = None
_model_lock = threading.RLock()


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise gr.Error(
            "GPU پیدا نشد. در Google Colab از Runtime → Change runtime type، "
            "گزینه T4 GPU را انتخاب کنید و دوباره اجرا کنید."
        )


def unload_models(except_model: str | None = None) -> None:
    """Keep only one large model resident to reduce Colab T4 out-of-memory errors."""
    global _chat_model, _chat_tokenizer, _image_pipe
    if except_model != "chat" and _chat_model is not None:
        del _chat_model
        _chat_model = None
        _chat_tokenizer = None
    if except_model != "image" and _image_pipe is not None:
        del _image_pipe
        _image_pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_chat_model() -> tuple[Any, Any]:
    global _chat_model, _chat_tokenizer
    require_cuda()
    with _model_lock:
        if _chat_model is None:
            unload_models(except_model="chat")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_ID)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                CHAT_MODEL_ID,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            _chat_tokenizer, _chat_model = tokenizer, model
    return _chat_model, _chat_tokenizer


def load_image_pipeline() -> Any:
    global _image_pipe
    require_cuda()
    with _model_lock:
        if _image_pipe is None:
            unload_models(except_model="image")
            _image_pipe = StableDiffusionXLPipeline.from_pretrained(
                IMAGE_MODEL_ID,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to("cuda")
            _image_pipe.enable_attention_slicing()
            _image_pipe.enable_vae_slicing()
            _image_pipe.enable_vae_tiling()
    return _image_pipe


def chat_stream(
    message: str,
    history: list[dict[str, str]] | None,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
):
    model, tokenizer = load_chat_model()
    messages: list[dict[str, str]] = []
    system = clean_text(system_prompt)
    if system:
        messages.append({"role": "system", "content": system})

    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role", "user")
            content = clean_text(turn.get("content", turn.get("text", "")))
            if role in {"user", "assistant", "system"} and content:
                messages.append({"role": role, "content": content})
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_text, assistant_text = clean_text(turn[0]), clean_text(turn[1])
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

    user_text = clean_text(message)
    if not user_text:
        yield "لطفاً پیام خود را وارد کنید."
        return
    messages.append({"role": "user", "content": user_text})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = inputs.to(device)
    streamer = TextIteratorStreamer(
        tokenizer, timeout=120.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_args = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": int(max_tokens),
        "do_sample": float(temperature) > 0,
        "temperature": max(float(temperature), 0.01),
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
    }
    thread = threading.Thread(target=model.generate, kwargs=generate_args, daemon=True)
    thread.start()
    accumulated = ""
    for piece in streamer:
        accumulated += piece
        yield accumulated


def generate_image(
    prompt: str,
    negative_prompt: str,
    ratio: str,
    steps: int,
    guidance: float,
    seed: float,
):
    try:
        positive, negative = build_image_prompts(prompt, negative_prompt)
        width, height = resolve_dimensions(ratio)
        pipe = load_image_pipeline()
        generator = None
        seed_value = int(seed)
        if seed_value >= 0:
            generator = torch.Generator(device="cuda").manual_seed(seed_value)
        result = pipe(
            prompt=positive,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator,
        )
        return result.images[0]
    except gr.Error:
        raise
    except Exception as exc:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(f"تولید تصویر ناموفق بود: {exc}") from exc


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="violet", secondary_hue="slate")
    with gr.Blocks(title="AI Studio | گفت‌وگو و تصویر", theme=theme) as demo:
        gr.Markdown(
            "# ✦ AI Studio\n"
            "محیط آزمایشی مدل‌های متن‌باز برای گفت‌وگو و تولید تصویر در Google Colab."
        )
        gr.Markdown(
            "نکته: اجرای مدل‌ها به GPU نیاز دارد. در Colab، ابتدا GPU را فعال کنید. "
            "به‌دلیل محدودیت حافظه T4، مدل گفت‌وگو و تصویر هم‌زمان در حافظه نگه‌داری نمی‌شوند."
        )
        with gr.Tab("گفت‌وگوی متنی"):
            system_prompt = gr.Textbox(
                label="دستورالعمل سیستم",
                value="You are a helpful, accurate assistant. Be clear about uncertainty.",
                lines=3,
            )
            chatbot = gr.ChatInterface(
                fn=chat_stream,
                type="messages",
                additional_inputs=[
                    system_prompt,
                    gr.Slider(64, 1536, value=768, step=64, label="حداکثر توکن پاسخ"),
                    gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="خلاقیت"),
                ],
                title=f"مدل متنی: {CHAT_MODEL_ID}",
                description="بارگذاری اولیه مدل ممکن است چند دقیقه طول بکشد.",
            )
        with gr.Tab("تولید تصویر"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="پرامپت تصویر",
                        placeholder="مثلاً: cinematic mountain landscape at sunrise, detailed, natural light",
                        lines=4,
                    )
                    negative = gr.Textbox(
                        label="پرامپت منفی اضافه (اختیاری)", lines=2
                    )
                    ratio = gr.Dropdown(
                        choices=list(ASPECT_RATIOS),
                        value="16:9 Landscape",
                        label="نسبت تصویر",
                    )
                    with gr.Row():
                        steps = gr.Slider(20, 35, value=IMAGE_STEPS_DEFAULT, step=1, label="Steps")
                        guidance = gr.Slider(4.0, 9.0, value=6.0, step=0.5, label="Guidance scale")
                    seed = gr.Number(value=-1, precision=0, label="Seed (عدد منفی یعنی تصادفی)")
                    generate = gr.Button("ساخت تصویر", variant="primary")
                with gr.Column():
                    output = gr.Image(label="خروجی", type="pil")
            generate.click(
                generate_image,
                inputs=[prompt, negative, ratio, steps, guidance, seed],
                outputs=output,
            )
            gr.Markdown(f"مدل تصویر: `{IMAGE_MODEL_ID}`")

        with gr.Accordion("راهنمای حافظه GPU", open=False):
            gr.Markdown(
                "- مدل‌ها در اولین اجرا دانلود می‌شوند و ممکن است چند دقیقه زمان ببرند.\n"
                "- اگر خطای کمبود حافظه دیدید، Runtime را Restart کنید و فقط یک بخش را اجرا کنید.\n"
                "- سهمیه و دسترسی GPU رایگان Colab تضمین‌شده نیست.\n"
                "- برای تغییر مدل‌ها، متغیرهای محیطی CHAT_MODEL_ID و IMAGE_MODEL_ID را تنظیم کنید."
            )
    return demo


if __name__ == "__main__":
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        share=os.getenv("GRADIO_SHARE", "true").lower() == "true",
        show_error=True,
    )
