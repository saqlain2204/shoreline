---
title: Shoreline
emoji: 🌊
colorFrom: blue
colorTo: yellow
sdk: static
pinned: false
license: mit
---

# An environment that scores land and water

The question is small. Give a latitude and a longitude, and answer with one word: land or water. The model never sees the label. A separate program picks the place, writes the prompt, and scores the reply. That program is the environment.

The card on the page is one episode with no clicks. `reset` starts it and returns an observation. `step` sends an action. The environment answers with a reward. An [OpenEnv](https://github.com/huggingface/OpenEnv) server exposes the same loop, so a trainer can call it without holding the atlas.

## Why the environment is there

The model starts as `Qwen/Qwen2.5-0.5B-Instruct`. It can follow a chat prompt. It does not contain this map. Before any update, the same question scored 47 percent on 200 balanced places, and it almost never said land.

The atlas is what makes a reply right or wrong. It is a grid of Natural Earth 1:110m coastlines and lakes, which are in the public domain. Each cell is a quarter of a degree. About two thirds of the cells are water. Lakes such as the Caspian Sea and the Great Lakes are water. A harbor on that simplified coast can fall in the neighboring cell, so the label is the atlas, not a navigational chart.

## Three tasks, one atlas

**Survey** is one step. The observation is the question. The action is free text. Reward is 1 when the last `land` or `water` in the reply matches the cell, and 0 otherwise.

```
Land or Water?
Latitude: 48.86
Longitude: 2.35
Reply with one word: land or water.
```

**Coast** starts inland. The agent may move north, south, east, or west, then say `here`. `here` on the shoreline scores 1. Anything else that ends the episode scores 0. A step that only moves scores 0 and the episode continues.

**Locate** drops the agent on a hidden place. The prompt has a coarse sketch of land and water, not the true coordinates. The agent can look north, south, east, or west, then commit with a latitude and a longitude. The reward falls with kilometres of error, and each look costs a little.

## Where reinforcement learning fits

Survey has one correct word, so the environment can show that word as well as score it. Coast and locate do not. A walk can be right in more than one sequence, and a locate guess is better when it is closer, not when it matches a single sentence. There the trainer only gets a number. The GRPO loop samples a few answers to the same observation, scores them in the environment, and updates a LoRA from which answers scored higher.

On the survey question that run collapsed to always saying water. A group of identical answers has no advantage, and the update also pulls the model back toward the base, which already prefers water. The adapter published with this project is not that run. It is the supervised fit: each step shows eight coordinates, half land and half water, and the target reply is the word the atlas would mark correct.

## What the weights learned

The published adapter is a LoRA of rank 8 on the 0.5B instruct model, trained for 400 steps on one T4. That is 3,200 places. The weights are at [vanishingradient/shoreline-land-water](https://huggingface.co/vanishingradient/shoreline-land-water).

The same prompt, before any update, scored 47 percent on 200 balanced places, with 5 percent of the land cells correct and every water cell correct. After training, a fresh 200 balanced places scored 75.5 percent: 61 percent of land cells and 92 percent of water cells. On 200 places drawn uniformly from the globe, the score was 81.5 percent: 72 percent on land and 86 percent on water.

The weights hold a rough continent shape. Open ocean and broad interiors are easier than a cell that sits on the 1:110m shore. Training draws land and water equally, because a model that always says water is right on about two thirds of a uniform sample and looks better than it is.

## On this page

The map takes a latitude and a longitude, or a click. It draws the atlas tile and shows the word the adapter gave for the one-degree cell that contains that place, next to the atlas label.

The code is at [github.com/saqlain2204/shoreline](https://github.com/saqlain2204/shoreline). One local place:

```bash
python -m shoreline.demo --lat 48.86 --lon 2.35 --answer land
```
