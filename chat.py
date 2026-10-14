"""Daredevil 8B chat interface, optimized for Colab T4."""
from __future__ import annotations
import os, threading, gc
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

MODEL_ID = os.getenv("CHAT_MODEL_ID", "mlabonne/Daredevil-8B-abliterated")
_model = _tokenizer = None
_lock = threading.RLock()

def load_model():
    global _model, _tokenizer
    if not torch.cuda.is_available():
        raise gr.Error("GPU در دسترس نیست. در Google Colab از Runtime → Change runtime type گزینه GPU را فعال کنید.")
    with _lock:
        if _model is None:
            gc.collect()
            torch.cuda.empty_cache()
            config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=config,
                device_map="auto", low_cpu_mem_usage=True)
            _tokenizer, _model = tokenizer, model
    return _model, _tokenizer

def clean_text(data):
    if data is None: return ""
    if isinstance(data, str): return data.strip()
    if isinstance(data, dict): return str(data.get("text", data.get("content", ""))).strip()
    if isinstance(data, (list, tuple)):
        return " ".join(filter(None, (clean_text(item) for item in data))).strip()
    return str(data).strip()

def chat_stream(message, history, system_prompt, max_tokens, temperature):
    user_text = clean_text(message)
    if not user_text:
        yield "لطفاً پیام خود را وارد کنید."
        return
    model, tokenizer = load_model()
    messages = []
    system_text = clean_text(system_prompt)
    if system_text: messages.append({"role": "system", "content": system_text})
    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role", "user")
            content = clean_text(turn.get("content", turn.get("text", "")))
            if role in {"system", "user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            for role, content in (("user", turn[0]), ("assistant", turn[1])):
                content = clean_text(content)
                if content: messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, timeout=120, skip_prompt=True, skip_special_tokens=True)
    kwargs = {**inputs, "streamer": streamer, "max_new_tokens": int(max_tokens),
        "do_sample": float(temperature) > 0, "top_p": 0.9, "pad_token_id": tokenizer.pad_token_id}
    if float(temperature) > 0: kwargs["temperature"] = float(temperature)
    threading.Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    result = ""
    for token in streamer:
        result += token
        yield result

with gr.Blocks(title="AI Studio | Daredevil Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ AI Studio · گفت‌وگوی متنی")
    gr.Markdown(f"مدل: `{MODEL_ID}` · بارگذاری مدل هنگام اولین پیام انجام می‌شود.")
    with gr.Accordion("تنظیمات مدل", open=False):
        system = gr.Textbox(label="دستورالعمل سیستم",
            value="You are a helpful, direct, and accurate assistant. Follow applicable safety rules.", lines=3)
        with gr.Row():
            max_tokens = gr.Slider(64, 1536, value=768, step=64, label="حداکثر توکن پاسخ")
            temperature = gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="Temperature")
    gr.ChatInterface(fn=chat_stream, type="messages", additional_inputs=[system, max_tokens, temperature],
        title="گفت‌وگو با هوش مصنوعی")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        share=os.getenv("GRADIO_SHARE", "true").lower() == "true", show_error=True)
