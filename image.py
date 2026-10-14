"""SDXL Pony image-generation interface for Colab T4."""
from __future__ import annotations
import gc, os
import gradio as gr
import torch
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

MODEL_ID = os.getenv("IMAGE_MODEL_ID", "John6666/ebara-pony-v1-sdxl")
VAE_ID = os.getenv("SDXL_VAE_ID", "madebyollin/sdxl-vae-fp16-fix")
PIPE = None
ASPECT_RATIOS = {
    "16:9 (Landscape - YouTube)": (1216, 832),
    "1:1 (Square)": (1024, 1024),
    "9:16 (Portrait - Shorts/Stories)": (832, 1216),
    "4:3 (Classic)": (1152, 864),
}
BASE_NEGATIVE = ("score_4, score_5, score_6, (worst quality, low quality:1.4), "
                 "bad anatomy, deformed limbs, bad hands, missing fingers, extra limbs, blurry")

def load_pipeline():
    global PIPE
    if not torch.cuda.is_available():
        raise gr.Error("GPU در دسترس نیست. در Colab، GPU را از تنظیمات Runtime فعال کنید.")
    if PIPE is None:
        gc.collect()
        torch.cuda.empty_cache()
        vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
        pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, vae=vae,
            torch_dtype=torch.float16, use_safetensors=True).to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        PIPE = pipe
    return PIPE

def generate_art(prompt, negative_prompt, style_tags, ratio, steps, cfg, seed):
    if not isinstance(prompt, str) or not prompt.strip():
        raise gr.Error("ابتدا پرامپت تصویر را وارد کنید.")
    try:
        width, height = ASPECT_RATIOS[ratio]
        positive = prompt.strip()
        tags = (style_tags or "").strip()
        if tags: positive = f"{tags}, {positive}"
        negative = BASE_NEGATIVE
        if negative_prompt and negative_prompt.strip(): negative = f"{negative}, {negative_prompt.strip()}"
        seed = int(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None
        return load_pipeline()(prompt=positive, negative_prompt=negative,
            num_inference_steps=int(steps), guidance_scale=float(cfg),
            width=width, height=height, generator=generator).images[0]
    except gr.Error:
        raise
    except Exception as exc:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        raise gr.Error(f"خطای پردازش GPU: {exc}") from exc

with gr.Blocks(title="AI Studio | SDXL Image", theme=gr.themes.Soft(primary_hue="purple")) as demo:
    gr.Markdown("# 🎨 AI Studio · تولید تصویر")
    gr.Markdown(f"مدل تصویر: `{MODEL_ID}` · بارگذاری مدل هنگام اولین اجرا انجام می‌شود.")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="پرامپت", lines=4, placeholder="سوژه، سبک، نورپردازی و ترکیب‌بندی را توصیف کنید.")
            negative = gr.Textbox(label="پرامپت منفی اضافه", lines=2)
            style_tags = gr.Textbox(label="تگ‌های سبک اختیاری", value="score_9, score_8_up, score_7_up")
            ratio = gr.Dropdown(choices=list(ASPECT_RATIOS), value=list(ASPECT_RATIOS)[0], label="ابعاد تصویر")
            with gr.Row():
                steps = gr.Slider(20, 35, value=25, step=1, label="Steps")
                cfg = gr.Slider(4.0, 8.0, value=6.0, step=0.5, label="CFG")
            seed = gr.Number(value=-1, precision=0, label="Seed (عدد منفی = تصادفی)")
            button = gr.Button("تولید تصویر", variant="primary")
        with gr.Column():
            output = gr.Image(label="تصویر تولیدشده", type="pil")
    button.click(generate_art, [prompt, negative, style_tags, ratio, steps, cfg, seed], output)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        share=os.getenv("GRADIO_SHARE", "true").lower() == "true", show_error=True)
