"""Convenience entry point for the separate chat and image interfaces.

Set AI_STUDIO_MODE=chat (default) or AI_STUDIO_MODE=image before launching.
The interfaces are deliberately run separately so large model weights do not
compete for the same Colab GPU memory.
"""
import os
import runpy

mode = os.getenv("AI_STUDIO_MODE", "chat").strip().lower()
if mode == "image":
    runpy.run_module("image", run_name="__main__")
elif mode == "chat":
    runpy.run_module("chat", run_name="__main__")
else:
    raise SystemExit("AI_STUDIO_MODE must be either 'chat' or 'image'.")
