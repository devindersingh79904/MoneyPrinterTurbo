"""Local OpenAI-compatible image server for MoneyPrinterTurbo's `openai_image` video source.

Serves POST /v1/images/generations with Stable Diffusion XL (diffusers) on the local GPU and
returns b64_json. The local LLM plans each shot from the task's script.json: a matching scene
prompt plus on-screen text (code, numbers, key phrase) drawn as a card over the image, so footage is generated on this PC instead of downloaded.

Run (from its own venv, see automation/README.md):
    python automation/local_image_server.py --port 7860

config.toml:
    video_source = "openai_image"
    openai_image_base_url = "http://127.0.0.1:7860/v1"
    openai_image_model = "sdxl"
    openai_image_size = "768x1344"      # portrait 9:16-ish; SDXL native bucket
"""
import argparse
import base64
import io
import json
import os
import threading
import time
from pathlib import Path

import requests
import torch
import uvicorn
from diffusers import AutoPipelineForText2Image
from fastapi import FastAPI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

MODEL_ID = os.environ.get("IMAGE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
PROMPT_LLM = os.environ.get("PROMPT_LLM", "qwen3:14b")
TASKS_DIR = Path(__file__).resolve().parents[1] / "storage" / "tasks"
NEGATIVE = ("text, letters, words, watermark, logo, signature, blurry, low quality, deformed, "
            "extra fingers, random people walking, street scene")

app = FastAPI()
_pipe = None
_lock = threading.Lock()
# script.json path -> {search term: {"prompt": SDXL prompt, "text": on-screen text}}
_prompt_cache: dict[str, dict] = {}
FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
CODE_FONT = os.environ.get("CARD_FONT", str(FONT_DIR / "consolab.ttf"))


def _current_task():
    """Newest task's script.json (written by the app before it asks for images), or None."""
    try:
        newest = max(TASKS_DIR.glob("*/script.json"), key=lambda p: p.stat().st_mtime)
    except ValueError:
        return None
    if time.time() - newest.stat().st_mtime > 3 * 3600:
        return None
    return newest


def _plan_shots(script_path: Path) -> dict:
    """Ask the local LLM once per video to plan every shot: a concrete SDXL scene that illustrates
    the narration, plus the short on-screen text (code, numbers, key phrase) for that moment."""
    key = str(script_path)
    if key in _prompt_cache:
        return _prompt_cache[key]
    data = json.loads(script_path.read_text(encoding="utf-8"))
    terms = [t for t in data.get("search_terms") or [] if isinstance(t, str)]
    subject = (data.get("params") or {}).get("video_subject", "")
    instructions = f"""You plan the shots of a short vertical explainer video. The narration script is
below; each search term marks one shot, in order. For EACH term return:

1. "prompt": a Stable Diffusion XL image prompt (25-45 words) that illustrates what the narration
   says at that point.
   - Describe concrete, drawable things: objects, setting, lighting, camera angle, style.
   - Abstract or technical ideas (math, code, bits, finance, science) become visual metaphors,
     e.g. glowing light bulbs on or off, a row of 8 switches, circuit boards, stacked blocks,
     a programmer's desk with screens.
   - Never ask for readable text, digits or words in the image (SDXL cannot spell; the real text
     is added separately). No unrelated people or street scenes.
   - One consistent style across all shots that suits the topic (e.g. "sleek 3D render, dark
     background, neon blue and orange glow" for tech, "photorealistic, natural light" for everyday).
2. "text": the on-screen text for that shot, exactly as a teacher would write it on the board
   while saying that part of the script. At most 2 lines (separate with \\n), each at most 22
   characters. Must be correct.
   - For technical topics PREFER code, a formula or a worked example over words, e.g.
     "n & (n - 1) == 0", "1000 & 0111\\n= 0000", "bin(5).count('1')\\n# 2", "n &= n - 1",
     "5 = 101\\n2 bits set", "O(k), k = set bits".
   - Only use a 2-5 word key phrase when there is nothing concrete to show (intro, outro,
     non-technical topics). Never just repeat the search term.

Reply with JSON only: {{"shots": [{{"prompt": "...", "text": "..."}}, ...]}} with exactly
{len(terms)} entries, same order as the search terms.

Video subject: {subject}
Script: {data.get("script", "")}
Search terms: {json.dumps(terms)}"""
    shots: dict = {}
    try:
        r = requests.post(f"{OLLAMA}/api/chat", timeout=300, json={
            "model": PROMPT_LLM, "stream": False, "think": False, "format": "json",
            "options": {"temperature": 0.3},
            "messages": [{"role": "user", "content": instructions}],
        })
        out = json.loads(r.json()["message"]["content"]).get("shots", [])
        for term, shot in zip(terms, out):
            if isinstance(shot, dict) and str(shot.get("prompt", "")).strip():
                shots[term] = {"prompt": str(shot["prompt"]).strip(),
                               "text": str(shot.get("text", "") or "").strip()}
                print(f"shot {term!r}: text={shots[term]['text']!r} prompt={shots[term]['prompt']}",
                      flush=True)
    except Exception as e:
        print(f"shot planning failed, falling back to plain terms: {e}", flush=True)
    shots["__subject__"] = subject
    _prompt_cache[key] = shots
    return shots


def _build_shot(term: str) -> tuple[str, str]:
    """(SDXL prompt, on-screen text) for one requested search term."""
    task = _current_task()
    if task is None:
        return f"{term}, high quality, detailed", ""
    shots = _plan_shots(task)
    if term in shots:
        return shots[term]["prompt"], shots[term]["text"]
    # Term not in this task's list (e.g. a retry with a rephrased term): still tie it to the subject.
    return (f"{term}, illustrating {shots.get('__subject__', '')}, high quality digital illustration",
            "")


def _font(size: int):
    for path in (CODE_FONT, str(FONT_DIR / "consola.ttf"), str(FONT_DIR / "segoeuib.ttf")):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_card(img: Image.Image, text: str) -> Image.Image:
    """Draw the on-screen text as a code card in the upper third (subtitles use the bottom).
    Kept inside a safe area because the app center-crops and slow-zooms the image."""
    lines = [ln.strip() for ln in text.replace("\\n", "\n").split("\n") if ln.strip()][:3]
    if not lines:
        return img
    img = img.convert("RGBA")
    w, h = img.size
    max_w = int(w * 0.74)  # safe width after crop + zoom
    size = int(w * 0.11)
    while size > 18:
        font = _font(size)
        widths = [font.getbbox(ln)[2] - font.getbbox(ln)[0] for ln in lines]
        if max(widths) <= max_w:
            break
        size -= 2
    line_h = int(size * 1.25)
    pad_x, pad_y = int(size * 0.7), int(size * 0.55)
    box_w = max(widths) + 2 * pad_x
    box_h = line_h * len(lines) + 2 * pad_y - int(size * 0.25)
    x0 = (w - box_w) // 2
    y0 = int(h * 0.20)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle((x0 + 6, y0 + 8, x0 + box_w + 6, y0 + box_h + 8), radius=size // 2,
                        fill=(0, 0, 0, 110))  # soft shadow
    d.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=size // 2,
                        fill=(13, 17, 23, 225), outline=(255, 166, 0, 255), width=max(3, size // 14))
    for i, ln in enumerate(lines):
        lw = widths[i]
        d.text(((w - lw) // 2, y0 + pad_y + i * line_h - font.getbbox(ln)[1] // 2), ln,
               font=font, fill=(240, 246, 252, 255))
    return Image.alpha_composite(img, overlay).convert("RGB")


class ImageRequest(BaseModel):
    prompt: str
    model: str | None = None
    size: str | None = "768x1344"
    n: int | None = 1
    response_format: str | None = "b64_json"


def _free_ollama_vram():
    """Unload Ollama models so SDXL and the LLM don't compete for 16 GB of VRAM."""
    try:
        for m in requests.get(f"{OLLAMA}/api/ps", timeout=5).json().get("models", []):
            requests.post(f"{OLLAMA}/api/generate",
                          json={"model": m["name"], "keep_alive": 0}, timeout=30)
    except Exception as e:
        print(f"could not unload ollama models: {e}")


def _get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        # Moves each sub-model to the GPU only while it runs, leaving VRAM for Whisper later.
        _pipe.enable_model_cpu_offload()
    return _pipe


def _size(size: str | None) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in (size or "768x1344").lower().split("x"))
    except ValueError:
        w, h = 768, 1344
    # SDXL needs multiples of 8; keep within ~1 MP.
    return max(512, w - w % 8), max(512, h - h % 8)


@app.post("/v1/images/generations")
def generate(req: ImageRequest):
    width, height = _size(req.size)
    with _lock:
        prompt, card_text = _build_shot(req.prompt.strip())
        _free_ollama_vram()
        pipe = _get_pipe()
        start = time.time()
        images = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE, width=width, height=height,
            num_inference_steps=25, guidance_scale=6.0,
            num_images_per_prompt=max(1, min(req.n or 1, 4)),
        ).images
        print(f"{len(images)} image(s) {width}x{height} in {time.time() - start:.1f}s: "
              f"{req.prompt!r} -> text={card_text!r} prompt={prompt!r}", flush=True)
        if card_text:
            images = [_draw_card(img, card_text) for img in images]
    data = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})
    return {"created": int(time.time()), "data": data}


@app.get("/v1/models")
def models():
    return {"data": [{"id": "sdxl", "object": "model"}]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--preload", action="store_true", help="download/load the model at startup")
    args = ap.parse_args()
    if args.preload:
        _get_pipe()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
