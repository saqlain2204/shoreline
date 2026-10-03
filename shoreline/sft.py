"""Fit a LoRA so the reply is the one word the atlas scores as correct."""

from __future__ import annotations

from pathlib import Path


def train_labels(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    steps: int = 400,
    batch_size: int = 8,
    lr: float = 1e-4,
    seed: int = 0,
    output_dir: str = "outputs/shoreline",
):
    """One balanced coordinate per row. The target text is ``land`` or ``water``."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from shoreline.sim import ShoreSim

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=lr,
    )
    sim = ShoreSim()
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for row in range(batch_size):
            outcome = sim.reset(
                seed=seed + step * batch_size + row,
                task="survey",
                include_image=False,
                balance=True,
            )
            label = sim._label
            messages = [
                {"role": "user", "content": outcome.prompt},
                {"role": "assistant", "content": label},
            ]
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": outcome.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(full, return_tensors="pt").to(device)
            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            labels = encoded["input_ids"].clone()
            labels[:, :prompt_len] = -100
            loss = model(**encoded, labels=labels).loss
            (loss / batch_size).backward()
            batch_loss += float(loss.detach()) / batch_size
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 25 == 0 or step == 0:
            print(f"step {step + 1}/{steps}  loss {batch_loss:.4f}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    print(f"Saved adapter to {destination}")
    return destination


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the one-word land/water reply")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default="outputs/shoreline")
    args = parser.parse_args()
    train_labels(steps=args.steps, batch_size=args.batch_size, output_dir=args.out)


if __name__ == "__main__":
    main()
