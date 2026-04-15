"""Gradio app: Gemma 4 E2B refusal removal comparison.

Inspired by: https://github.com/TrevorS/gemma-4-abliteration
Side-by-side comparison of original vs. uncensored (refusal directions removed) model.
"""

import gc
import os
import time
from pathlib import Path

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from root-level helpers
from config import (
    DTYPE, N_LAYERS, TARGET_MODULES, TEST_PROMPTS,
    get_device, get_text_layers, clear_mps_cache,
)
from evaluate import generate_response, classify_response
from abliterate import abliterate_model

MAX_NEW_TOKENS = 150  # Shorter responses for faster inference

state = {
    "model": None, "tokenizer": None, "device": None,
    "original_weights": {}, "uncensored_weights": {}, "ready": False,
}

def _get_target(layer, module_name):
    if module_name == "o_proj":
        return layer.self_attn.o_proj
    elif module_name == "down_proj":
        return layer.mlp.down_proj
    raise ValueError(f"Unknown module: {module_name}")

def _backup_weights(model) -> dict:
    """Clone all 70 target weight matrices to CPU."""
    layers = get_text_layers(model)
    backup = {}
    for i in range(N_LAYERS):
        for mod in TARGET_MODULES:
            key = f"layer.{i}.{mod}"
            target = _get_target(layers[i], mod)
            backup[key] = target.weight.data.detach().clone().cpu()
    return backup

def swap_weights(weights_dict: dict):
    """Swap model weights in-place from a backup dict."""
    layers = get_text_layers(state["model"])
    for i in range(N_LAYERS):
        for mod in TARGET_MODULES:
            key = f"layer.{i}.{mod}"
            target = _get_target(layers[i], mod)
            target.weight.data.copy_(weights_dict[key].to(target.weight.device))
    clear_mps_cache()

def initialize():
    """Load model, backup weights, remove refusals, backup again."""
    device = get_device()
    state["device"] = device
    print(f"Device: {device}")

    model_id = "google/gemma-4-E2B-it"
    hf_token = os.environ.get("HF_TOKEN")
    print(f"Loading from HF Hub: {model_id}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    print("Loading model (this may take a few minutes on first run)...")
    t0 = time.time()

    if device.type == "cuda":
        # Use SDPA attention for ~20% faster inference
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=DTYPE, device_map="auto",
            attn_implementation="sdpa", token=hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE, token=hf_token)
        state["device"] = torch.device("cpu")
        device = state["device"]

    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    state["model"] = model
    state["tokenizer"] = tokenizer

    print("Backing up original weights (70 matrices)...")
    state["original_weights"] = _backup_weights(model)

    # Load pre-computed refusal directions
    directions_path = Path(__file__).parent / "cache" / "directions.pt"
    if not directions_path.exists():
        raise FileNotFoundError(f"directions.pt not found at {directions_path}")
    cached = torch.load(directions_path, weights_only=True)
    refusal_dirs = cached["directions"]
    print(f"Loaded refusal directions: {refusal_dirs.shape}")

    print("Applying refusal removal (uncensoring)...")
    t0 = time.time()
    abliterate_model(model, refusal_dirs, device)
    print(f"Refusal removal done in {time.time() - t0:.1f}s")

    print("Backing up uncensored weights...")
    state["uncensored_weights"] = _backup_weights(model)

    state["ready"] = True
    gc.collect()
    clear_mps_cache()
    print("Ready.")

_BS = "display:inline-flex;align-items:center;gap:8px;padding:6px 14px 6px 10px;border-radius:4px;font-family:'Space Mono',monospace;font-size:10px;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;"
_DOT = "width:7px;height:7px;border-radius:50%;flex-shrink:0;display:inline-block;"

_BADGES = {
    "refused": f'<span style="{_BS}background:rgba(242,95,76,0.12);border:1px solid rgba(242,95,76,0.32);color:#F25F4C;"><span style="{_DOT}background:#F25F4C;"></span>Refused</span>',
    "complied": f'<span style="{_BS}background:rgba(167,201,87,0.11);border:1px solid rgba(167,201,87,0.30);color:#A7C957;"><span style="{_DOT}background:#A7C957;"></span>Complied</span>',
    "disclaimer_but_complied": f'<span style="{_BS}background:rgba(232,197,71,0.11);border:1px solid rgba(232,197,71,0.30);color:#E8C547;"><span style="{_DOT}background:#E8C547;"></span>Disclaimer</span>',
}

def format_badge(cls: str) -> str:
    return _BADGES.get(cls, f'<span style="font-family:Space Mono,monospace;font-size:10px;color:#7A788F">{cls}</span>')

def format_summary(orig_cls: str, uncensored_cls: str) -> str:
    ob = _BADGES.get(orig_cls, f'<span style="color:#7A788F;font-family:Space Mono,monospace;font-size:10px">{orig_cls}</span>')
    ab = _BADGES.get(uncensored_cls, f'<span style="color:#7A788F;font-family:Space Mono,monospace;font-size:10px">{uncensored_cls}</span>')
    return f'<div class="summary-bar"><span class="sum-label">Outcome</span><span class="sum-col-name">Safety-Trained</span>{ob}<span class="sum-vs">vs</span>{ab}<span class="sum-col-name">Uncensored</span></div>'

def compare(prompt: str):
    if not state["ready"]:
        empty = format_summary("", "")
        return "Model not loaded yet.", "", "Model not loaded yet.", "", empty

    prompt = prompt.strip()
    if not prompt:
        empty = format_summary("", "")
        return "Please enter a prompt.", "", "Please enter a prompt.", "", empty

    device = state["device"]
    model = state["model"]
    tokenizer = state["tokenizer"]

    swap_weights(state["original_weights"])
    orig_response = generate_response(model, tokenizer, prompt, device, max_new_tokens=MAX_NEW_TOKENS)
    orig_class = classify_response(orig_response)

    swap_weights(state["uncensored_weights"])
    uncensored_response = generate_response(model, tokenizer, prompt, device, max_new_tokens=MAX_NEW_TOKENS)
    uncensored_class = classify_response(uncensored_response)

    gc.collect()
    clear_mps_cache()

    return (
        orig_response, format_badge(orig_class),
        uncensored_response, format_badge(uncensored_class),
        format_summary(orig_class, uncensored_class),
    )

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400;1,600&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');
:root {
    --void: #0C0B14; --surface: #13121E; --raised: #1C1B2B; --border: #26244080; --border-s: #262440;
    --text: #EEEDF8; --muted: #74728A; --dim: #44425A; --accent: #FF8906; --ag: rgba(255,137,6,0.14);
    --green: #A7C957; --red: #F25F4C; --yellow: #E8C547; --fd: 'Cormorant Garamond', Georgia, serif;
    --fm: 'Space Mono', 'Courier New', monospace;
}
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container, gradio-app {
    background: var(--void) !important; color: var(--text) !important; font-family: var(--fm) !important;
}
.gradio-container { max-width: 1340px !important; margin: 0 auto !important; }
.block, .gap, .contain, .padded { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
.hdr { padding: 40px 0 28px; border-bottom: 1px solid var(--border-s); margin-bottom: 32px; }
.hdr-eye { font-size: 10px; letter-spacing: 0.30em; text-transform: uppercase; color: var(--accent); margin-bottom: 12px; }
.hdr-title { font-family: var(--fd); font-size: clamp(2.2rem,5vw,4rem); font-weight: 300; line-height: 1.06; letter-spacing: -0.01em; color: var(--text); margin: 0 0 10px; }
.hdr-title em { font-style: italic; color: var(--accent); }
.hdr-tagline { font-size: 12px; color: var(--muted); line-height: 1.9; max-width: 540px; margin-bottom: 16px; }
.hdr-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.hdr-chip { font-size: 9px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--dim); border: 1px solid var(--border-s); border-radius: 2px; padding: 3px 9px; }
.hdr-chip.hi { color: var(--accent); border-color: rgba(255,137,6,0.30); background: var(--ag); }
#input-row { background: var(--raised) !important; border: 1px solid var(--border-s) !important; border-radius: 8px !important; padding: 20px !important; margin-bottom: 0 !important; box-shadow: 0 4px 24px rgba(0,0,0,0.28) !important; }
textarea { background: var(--surface) !important; border: 1px solid var(--border-s) !important; color: var(--text) !important; font-family: var(--fm) !important; font-size: 14px !important; line-height: 1.8 !important; border-radius: 5px !important; padding: 14px 16px !important; transition: border-color 0.18s, box-shadow 0.18s !important; caret-color: var(--accent) !important; }
textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--ag) !important; outline: none !important; }
textarea::placeholder { color: var(--dim) !important; }
textarea[readonly] { background: var(--raised) !important; }
.label-wrap > span, label > span { font-family: var(--fm) !important; font-size: 9px !important; letter-spacing: 0.20em !important; text-transform: uppercase !important; color: var(--dim) !important; font-weight: 400 !important; }
button.lg.primary, button.primary { background: var(--accent) !important; color: #0A0913 !important; border: none !important; border-radius: 5px !important; font-family: var(--fm) !important; font-size: 12px !important; font-weight: 700 !important; letter-spacing: 0.14em !important; text-transform: uppercase !important; transition: opacity 0.14s, transform 0.14s, box-shadow 0.14s !important; min-height: 48px !important; }
button.lg.primary:hover, button.primary:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; box-shadow: 0 8px 28px var(--ag) !important; }
button.lg.primary:disabled, button.primary:disabled { background: var(--dim) !important; color: var(--muted) !important; }
button:not(.primary) { background: var(--raised) !important; color: var(--muted) !important; border: 1px solid var(--border-s) !important; border-radius: 4px !important; font-family: var(--fm) !important; font-size: 10px !important; }
.chips-label { font-size: 9px; letter-spacing: 0.20em; text-transform: uppercase; color: var(--dim); margin: 10px 0 6px; }
#examples-chips { padding: 0 !important; margin-bottom: 16px !important; }
.examples-holder { border: none !important; background: transparent !important; }
.examples-holder .dataset { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
.examples-holder .dataset table { display: flex !important; flex-wrap: wrap !important; gap: 8px !important; border: none !important; background: transparent !important; }
.examples-holder .dataset thead { display: none !important; }
.examples-holder .dataset tbody { display: contents !important; }
.examples-holder .dataset tr { display: contents !important; }
.examples-holder .dataset td { display: inline-flex !important; align-items: center !important; font-family: var(--fm) !important; font-size: 10px !important; letter-spacing: 0.04em !important; color: var(--muted) !important; border: 1px solid var(--border-s) !important; background: var(--surface) !important; border-radius: 4px !important; padding: 5px 12px !important; cursor: pointer !important; transition: all 0.14s ease !important; white-space: nowrap !important; max-width: 260px !important; overflow: hidden !important; text-overflow: ellipsis !important; }
.examples-holder .dataset td:hover { color: var(--text) !important; border-color: var(--accent) !important; background: var(--ag) !important; }
.summary-bar { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; background: var(--raised); border: 1px solid var(--border-s); border-radius: 6px; padding: 12px 18px; margin: 16px 0; animation: fadeUp 0.3s ease forwards; }
.sum-label { font-size: 9px; letter-spacing: 0.22em; text-transform: uppercase; color: var(--dim); margin-right: 4px; flex-shrink: 0; }
.sum-vs { font-size: 11px; color: var(--dim); margin: 0 2px; }
.sum-col-name { font-size: 9px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
#col-orig, #col-abl { display: flex !important; flex-direction: column !important; align-items: stretch !important; justify-content: flex-start !important; border-radius: 8px !important; padding: 18px !important; }
#col-orig { background: rgba(242,95,76,0.04) !important; border: 1px solid rgba(242,95,76,0.14) !important; }
#col-abl { background: rgba(167,201,87,0.04) !important; border: 1px solid rgba(167,201,87,0.14) !important; }
#col-orig > *, #col-abl > * { flex-shrink: 0 !important; }
#col-orig > div:last-child, #col-abl > div:last-child { flex: 1 1 auto !important; display: flex !important; flex-direction: column !important; }
#col-orig > div:last-child textarea, #col-abl > div:last-child textarea { flex: 1 1 auto !important; height: 100% !important; background: rgba(255,255,255,0.03) !important; min-height: 200px !important; }
.ph { display: flex; align-items: center; gap: 9px; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid var(--border-s); }
.ph-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.ph-dot.orig { background: var(--red); box-shadow: 0 0 6px rgba(242,95,76,0.5); }
.ph-dot.abl { background: var(--green); box-shadow: 0 0 6px rgba(167,201,87,0.5); }
.ph-name { font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text); font-weight: 700; }
.ph-tag { margin-left: auto; font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--dim); border: 1px solid var(--border-s); padding: 2px 8px; border-radius: 2px; }
.badge-row { min-height: 38px !important; display: flex !important; align-items: center !important; padding: 4px 0 10px !important; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
.result-reveal { animation: fadeUp 0.32s ease forwards; }
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-s); border-radius: 2px; }
footer { display: none !important; }
"""

THEME = gr.themes.Base(primary_hue="orange", neutral_hue="slate", font=gr.themes.GoogleFont("Space Mono"), font_mono=gr.themes.GoogleFont("Space Mono"))

_JS = """<script>
(function() {
  function watchOutputs() {
    ['col-orig', 'col-abl'].forEach(function(id) {
      var col = document.getElementById(id);
      if (!col) return;
      var ta = col.querySelector('textarea');
      if (!ta) return;
      var prev = ta.value;
      setInterval(function() {
        if (ta.value !== prev && ta.value.length > 0) {
          prev = ta.value;
          col.classList.remove('result-reveal');
          void col.offsetWidth;
          col.classList.add('result-reveal');
        }
      }, 400);
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() { setTimeout(watchOutputs, 1500); });
  } else {
    setTimeout(watchOutputs, 1500);
  }
})();
</script>"""

def build_ui():
    device_label = str(state.get("device", "—"))
    with gr.Blocks(title="Refusal Removal — Gemma 4 E2B") as demo:
        gr.HTML(_JS)
        gr.HTML(f"""
        <div class="hdr">
          <div class="hdr-eye">Research Instrument</div>
          <h1 class="hdr-title">Gemma 4 E2B &mdash; <em>Refusal Removal</em> Study</h1>
          <p class="hdr-tagline">
            Compare baseline model vs. uncensored model behavior side-by-side.<br>
            Refusal directions surgically removed from 70 weight matrices via norm-preserving rank-1 projection.<br>
            <small style="color:#7A788F">Inspired by <a href="https://github.com/TrevorS/gemma-4-abliteration" style="color:#FF8906">github.com/TrevorS/gemma-4-abliteration</a></small>
          </p>
          <div class="hdr-meta">
            <span class="hdr-chip hi">{device_label}</span>
            <span class="hdr-chip">fp16</span>
            <span class="hdr-chip">35 layers</span>
            <span class="hdr-chip">weight-swap mode</span>
          </div>
        </div>
        """)

        with gr.Row(equal_height=True, elem_id="input-row"):
            prompt_input = gr.Textbox(label="Your prompt", placeholder="Ask anything…", lines=3, scale=7)
            submit_btn = gr.Button("Run →", variant="primary", scale=1, min_width=120)

        gr.HTML('<div class="chips-label">Try an example</div>')
        gr.Examples(examples=[[p] for p in TEST_PROMPTS[:6]], inputs=[prompt_input], label=None, elem_id="examples-chips")

        summary_out = gr.HTML(value="", elem_id="summary-bar")

        with gr.Row(elem_id="panels-row"):
            with gr.Column(elem_id="col-orig"):
                gr.HTML('<div class="ph"><div class="ph-dot orig"></div><span class="ph-name"> Gemma 4 E2B Baseline Model</span><span class="ph-tag">Original</span></div>')
                orig_badge = gr.HTML(elem_classes=["badge-row"])
                orig_output = gr.Textbox(label="Response", lines=16, interactive=False)

            with gr.Column(elem_id="col-abl"):
                gr.HTML('<div class="ph"><div class="ph-dot abl"></div><span class="ph-name">Uncensored</span><span class="ph-tag">Refusals Removed</span></div>')
                abl_badge = gr.HTML(elem_classes=["badge-row"])
                abl_output = gr.Textbox(label="Response", lines=16, interactive=False)

        _outputs = [orig_output, orig_badge, abl_output, abl_badge, summary_out]
        submit_btn.click(fn=compare, inputs=[prompt_input], outputs=_outputs)
        prompt_input.submit(fn=compare, inputs=[prompt_input], outputs=_outputs)

    return demo

initialize()
demo = build_ui()
demo.queue()
demo.launch(theme=THEME, css=CSS)
