"""Group Relative Policy Optimization for the land-or-water question.

The update is the clipped GRPO objective: a group of answers to the same
coordinate share one advantage, and a KL term keeps the model near its base.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def group_advantages(rewards: list[float], group_size: int) -> list[float]:
    """Standardize rewards inside each group. A tied group gets advantage 0."""
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if len(rewards) % group_size != 0:
        raise ValueError("rewards must divide into complete groups")
    advantages: list[float] = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start : start + group_size]
        mean = sum(group) / len(group)
        variance = sum((item - mean) ** 2 for item in group) / len(group)
        scale = variance ** 0.5
        if scale < 1e-6:
            advantages.extend(0.0 for _ in group)
        else:
            advantages.extend((item - mean) / scale for item in group)
    return advantages


def latest_checkpoint(output_dir: Path) -> tuple[int, Path] | None:
    """Return the completed step and folder of the newest checkpoint-* directory."""
    if not output_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            continue
        step = int(suffix)
        if best is None or step > best[0]:
            best = (step, path)
    return best


def resolve_hub_repo(hub_repo: str | None) -> str | None:
    """Use an explicit repo id, or the signed-in user plus shoreline-land-water."""
    if hub_repo:
        return hub_repo
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return None
    from huggingface_hub import whoami

    name = whoami(token=token).get("name")
    if not name:
        return None
    return f"{name}/shoreline-land-water"


def _push_folder(
    path: Path,
    repo_id: str,
    message: str,
    private: bool,
    path_in_repo: str = ".",
    ignore_patterns: list[str] | None = None,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        commit_message=message,
        ignore_patterns=ignore_patterns,
    )


def train_survey(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    steps: int = 30,
    group_size: int = 4,
    lr: float = 1e-5,
    kl_coef: float = 0.02,
    clip_eps: float = 0.2,
    seed: int = 0,
    max_new_tokens: int = 6,
    output_dir: str = "outputs/shoreline",
    task: str = "survey",
    save_steps: int = 5,
    hub_repo: str | None = None,
    hub_private: bool = True,
    resume: bool = True,
):
    """Train a small instruct model on balanced land/water questions.

    Returns the per-step mean reward. A LoRA adapter is written every
    ``save_steps`` and, when a Hugging Face token is available, pushed to the Hub.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from shoreline.sim import ShoreSim

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    destination = Path(output_dir)
    found = latest_checkpoint(destination) if resume else None
    completed = 0
    history: list[float] = []
    if found is not None:
        completed, ckpt = found
        state_path = ckpt / "trainer_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        if state.get("model_name") not in (None, model_name):
            completed = 0
            found = None
        else:
            history = [float(item) for item in state.get("history", [])]
            model = PeftModel.from_pretrained(base, ckpt, is_trainable=True)
            print(f"Resumed from {ckpt} at step {completed}")
    if found is None:
        model = get_peft_model(
            base,
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
    if found is not None and (ckpt / "optimizer.pt").is_file():
        optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location=device, weights_only=True))
    repo_id = resolve_hub_repo(hub_repo)
    if repo_id:
        print(f"Hub repo {repo_id}")
    else:
        print("Hub push skipped: set HF_TOKEN or pass hub_repo")
    sim = ShoreSim()

    def completion_logprobs(input_ids: torch.Tensor, prompt_len: int):
        logits = model(input_ids=input_ids).logits[:, :-1, :]
        target = input_ids[:, 1:]
        chosen = torch.log_softmax(logits.float(), dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        positions = torch.arange(chosen.shape[1], device=chosen.device)
        mask = positions.unsqueeze(0) >= (prompt_len - 1)
        mask = mask & target.ne(tokenizer.pad_token_id)
        return chosen, mask

    for step in range(completed, steps):
        outcome = sim.reset(
            seed=seed + step,
            task=task,
            include_image=False,
            balance=task == "survey",
            max_moves=4,
            step_deg=10.0,
            span_deg=30.0,
        )
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": outcome.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = outcome.prompt + "\n"
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = int(encoded["input_ids"].shape[1])
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id,
            )
        texts = [
            tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
            for row in generated
        ]
        rewards = []
        for text in texts:
            sim.reset(
                task="survey",
                latitude=outcome.latitude,
                longitude=outcome.longitude,
                include_image=False,
            )
            rewards.append(float(sim.step(text).reward or 0.0))
        advantages = torch.tensor(
            group_advantages(rewards, group_size),
            device=device,
            dtype=torch.float32,
        )
        model.train()
        with torch.no_grad(), model.disable_adapter():
            ref_logprobs, token_mask = completion_logprobs(generated, prompt_len)
        new_logprobs, token_mask = completion_logprobs(generated, prompt_len)
        old_logprobs = new_logprobs.detach()
        new_seq = (new_logprobs * token_mask).sum(dim=1)
        old_seq = (old_logprobs * token_mask).sum(dim=1)
        ratio = torch.exp(new_seq - old_seq)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        kl_tokens = torch.exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1.0
        kl = (kl_tokens * token_mask).sum() / token_mask.sum().clamp_min(1.0)
        loss = policy_loss + kl_coef * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        mean_reward = sum(rewards) / len(rewards)
        history.append(mean_reward)
        print(
            f"step {step + 1}/{steps}  reward {mean_reward:.2f}  "
            f"loss {float(loss.detach()):.4f}  answers {texts}"
        )
        done = step + 1
        if done == steps or (save_steps > 0 and done % save_steps == 0):
            _save_checkpoint(
                model,
                tokenizer,
                destination,
                optimizer,
                completed_steps=done,
                history=history,
                model_name=model_name,
                repo_id=repo_id,
                hub_private=hub_private,
            )

    print(f"Saved adapter to {destination}")
    return history


def train_locate(**kwargs):
    """GRPO on the hidden-drop task. The model sees an atlas sketch, not the coordinate."""
    kwargs.setdefault("task", "locate")
    kwargs.setdefault("max_new_tokens", 24)
    kwargs.setdefault("output_dir", "outputs/locate")
    return train_survey(**kwargs)


def _save_checkpoint(
    model,
    tokenizer,
    destination: Path,
    optimizer,
    completed_steps: int,
    history: list[float],
    model_name: str,
    repo_id: str | None,
    hub_private: bool,
) -> None:
    import torch

    folder = destination / f"checkpoint-{completed_steps}"
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    torch.save(optimizer.state_dict(), folder / "optimizer.pt")
    state = {
        "completed_steps": completed_steps,
        "history": history,
        "model_name": model_name,
    }
    (folder / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    (destination / "README.md").write_text(_model_card(model_name), encoding="utf-8")
    print(f"Checkpoint {folder}")
    if not repo_id:
        return
    try:
        _push_folder(
            folder,
            repo_id,
            f"checkpoint {completed_steps}",
            hub_private,
            path_in_repo=f"checkpoints/checkpoint-{completed_steps}",
        )
        _push_folder(
            destination,
            repo_id,
            f"adapter at step {completed_steps}",
            hub_private,
            ignore_patterns=["checkpoint-*"],
        )
        print(f"Pushed to https://huggingface.co/{repo_id}")
    except Exception as exc:
        print(f"Hub push failed: {type(exc).__name__}: {exc}")


def _model_card(model_name: str) -> str:
    return f"""---
base_model: {model_name}
library_name: peft
tags:
- peft
- lora
- grpo
- shoreline
---

# Shoreline land-or-water adapter

LoRA adapter trained with GRPO to answer "Land or Water?" from a latitude and longitude.
The base model is [{model_name}](https://huggingface.co/{model_name}).
Checkpoints live under `checkpoints/`.
"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a small model on land/water with GRPO")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", choices=("survey", "locate"), default="survey")
    parser.add_argument("--out", default="outputs/grpo")
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--hub-repo", default=None, help="Hugging Face repo id. Defaults to <user>/shoreline-land-water when HF_TOKEN is set.")
    parser.add_argument("--hub-public", action="store_true")
    args = parser.parse_args()
    output = "outputs/locate" if args.task == "locate" and args.out == "outputs/grpo" else args.out
    train_survey(
        model_name=args.model,
        steps=args.steps,
        group_size=args.group_size,
        lr=args.lr,
        seed=args.seed,
        output_dir=output,
        save_steps=args.save_steps,
        hub_repo=args.hub_repo,
        hub_private=not args.hub_public,
        task=args.task,
        max_new_tokens=24 if args.task == "locate" else 6,
    )


if __name__ == "__main__":
    main()
