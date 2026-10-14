"""Unified web application for AI Studio.

One Gradio website exposes both original models. To fit common Colab GPUs,
the app unloads the other model when the user switches between chat and images.
"""
from __future__ import annotations

import gc
import os

import gradio as gr
import torch

import chat as chat_module
import image as image_module

CHAT_MODEL_ID = chat_module.MODEL_ID
IMAGE_MODEL_ID = image_module.MODEL_ID


def _clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def chat_with_memory_management(message, history, system_prompt, max_tokens, temperature):
    # Avoid holding both large model pipelines on the GPU at the same time.
    image_module.unload_pipeline()
    _clear_cuda()
    yield from chat_module.chat_stream(message, history, system_prompt, max_tokens, temperature)


def generate_with_memory_management(prompt, negative_prompt, style_tags, ratio, steps, cfg, seed):
    chat_module.unload_model()
    _clear_cuda()
    return image_module.generate_art(prompt, negative_prompt, style_tags, ratio, steps, cfg, seed)


with gr.Blocks(title="AI Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✨ AI Studio")
    gr.Markdown(
        "یک محیط وب برای گفت‌وگوی هوش مصنوعی و ساخت تصویر. "
        "مدل‌ها همان مدل‌های اصلی پروژه هستند و هنگام اولین استفاده بارگذاری می‌شوند."
    )
    with gr.Row():
        gr.Markdown(f"**مدل چت:** `{CHAT_MODEL_ID}`")
        gr.Markdown(f"**مدل تصویر:** `{IMAGE_MODEL_ID}`")
    with gr.Tabs():
        with gr.Tab("🤖 گفت‌وگوی هوش مصنوعی"):
            gr.Markdown("پاسخ‌ها به‌صورت زنده نمایش داده می‌شوند. هنگام رفتن به چت، مدل تصویر از حافظه GPU آزاد می‌شود.")
            with gr.Accordion("تنظیمات چت", open=False):
                system = gr.Textbox(
                    label="دستورالعمل سیستم",
                    value="You are a helpful, direct, and accurate assistant. Follow applicable safety rules.",
                    lines=3,
                )
                with gr.Row():
                    max_tokens = gr.Slider(64, 1536, value=768, step=64, label="حداکثر توکن پاسخ")
                    temperature = gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="Temperature")
            gr.ChatInterface(
                fn=chat_with_memory_management,
                type="messages",
                additional_inputs=[system, max_tokens, temperature],
                title="گفت‌وگو با Daredevil 8B",
            )
        with gr.Tab("🎨 ساخت تصویر"):
            gr.Markdown("توضیحات تصویر را بنویس و «تولید تصویر» را بزن. هنگام ساخت تصویر، مدل چت از حافظه GPU آزاد می‌شود.")
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="پرامپت تصویر",
                        lines=4,
                        placeholder="سوژه، سبک، نورپردازی و ترکیب‌بندی را توصیف کن.",
                    )
                    negative = gr.Textbox(label="پرامپت منفی اضافه", lines=2)
                    style_tags = gr.Textbox(
                        label="تگ‌های سبک اختیاری",
                        value="score_9, score_8_up, score_7_up",
                    )
                    ratio = gr.Dropdown(
                        choices=list(image_module.ASPECT_RATIOS),
                        value=list(image_module.ASPECT_RATIOS)[0],
                        label="نسبت تصویر",
                    )
                    with gr.Row():
                        steps = gr.Slider(20, 35, value=25, step=1, label="مراحل تولید")
                        cfg = gr.Slider(4.0, 8.0, value=6.0, step=0.5, label="CFG")
                    seed = gr.Number(value=-1, precision=0, label="Seed (عدد منفی = تصادفی)")
                    generate = gr.Button("✨ تولید تصویر", variant="primary")
                with gr.Column():
                    output = gr.Image(label="تصویر تولیدشده", type="pil")
            generate.click(
                fn=generate_with_memory_management,
                inputs=[prompt, negative, style_tags, ratio, steps, cfg, seed],
                outputs=output,
            )
    gr.Markdown(
        "⚠️ اجرای مدل‌ها به GPU سازگار و حافظه کافی نیاز دارد. اولین اجرا ممکن است به‌دلیل دانلود مدل‌ها طول بکشد."
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        share=os.getenv("GRADIO_SHARE", "true").lower() == "true",
        show_error=True,
    )
