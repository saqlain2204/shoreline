"""Ask the Shoreline adapter whether a coordinate is land or water."""

from __future__ import annotations

import tempfile

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from shoreline.sim import ShoreSim

ADAPTER = "vanishingradient/shoreline-land-water"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
PLACES = {
    "Paris": (48.86, 2.35),
    "Sahara": (23.0, 12.0),
    "Pacific": (0.0, -160.0),
    "Caspian": (41.5, 50.5),
}

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
try:
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
except TypeError:
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()
sim = ShoreSim()


def reply(prompt: str) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=6, do_sample=False)
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def ask(latitude: float, longitude: float):
    outcome = sim.reset(
        task="survey",
        latitude=float(latitude),
        longitude=float(longitude),
        include_image=True,
    )
    answer = reply(outcome.prompt)
    scored = sim.step(answer)
    truth = sim.world.label(float(latitude), float(longitude))
    note = "Matches the atlas." if scored.reward else "Does not match the atlas."
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(outcome.image_png)
        path = handle.name
    return path, answer, truth, note


def go_to(name: str):
    latitude, longitude = PLACES[name]
    image, answer, truth, note = ask(latitude, longitude)
    return latitude, longitude, image, answer, truth, note


with gr.Blocks(title="Shoreline") as demo:
    gr.Markdown(
        """
# Shoreline

Type a latitude and longitude. The tile is drawn from public-domain coastlines.
The model answers with one word, land or water.

On 200 fresh balanced places the published adapter scores 75.5 percent.
Before training, the same prompt scored 47 percent and almost never said land.
        """.strip()
    )
    with gr.Row():
        latitude = gr.Number(label="Latitude", value=48.86)
        longitude = gr.Number(label="Longitude", value=2.35)
    with gr.Row():
        ask_button = gr.Button("Ask", variant="primary")
        preset_buttons = [(name, gr.Button(name)) for name in PLACES]
    image = gr.Image(label="Atlas tile", type="filepath")
    answer = gr.Textbox(label="Model")
    truth = gr.Textbox(label="Atlas")
    note = gr.Textbox(label="Score")
    outputs = [image, answer, truth, note]
    ask_button.click(ask, [latitude, longitude], outputs)
    for name, button in preset_buttons:
        button.click(
            lambda name=name: go_to(name),
            outputs=[latitude, longitude, *outputs],
        )

if __name__ == "__main__":
    demo.launch()
