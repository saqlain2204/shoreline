---
title: Shoreline
emoji: 🌊
colorFrom: blue
colorTo: yellow
sdk: static
pinned: false
license: mit
---

# Teaching a small model the shape of the coast

Shoreline asks a plain question. Give a model a latitude and a longitude, and ask whether that place is land or water. The answer is one word. The score is whether that word matches an atlas.

The atlas is a grid made from Natural Earth 1:110m coastlines and lakes, which are in the public domain. Each cell is a quarter of a degree. A harbor on that simplified coast can fall in the neighboring cell, so the label is the atlas, not a navigational chart. Lakes such as the Caspian Sea and the Great Lakes are water.

## The episode

A survey episode is one step.

```
Land or Water?
Latitude: 48.86
Longitude: 2.35
Reply with one word: land or water.
```

The label is not in the prompt. Reward is 1 when the last `land` or `water` in the reply matches the cell, and 0 otherwise. Training draws land and water equally. On a uniform sample of the globe, water is the larger share, so a model that always says water looks better than it is.

Two other episodes sit on the same atlas. Coast starts inland and asks the model to walk north, south, east, or west until it says `here` on the shoreline. Locate hides the coordinate, shows a coarse sketch of the view, and scores a guessed latitude and longitude by distance.

## Training

The published adapter is a LoRA on `Qwen/Qwen2.5-0.5B-Instruct`, rank 8, on a single T4. Each step shows eight coordinates, half land and half water in expectation, and the target reply is the one word that would score 1. Four hundred steps cover 3,200 places. The weights are the adapter at [vanishingradient/shoreline-land-water](https://huggingface.co/vanishingradient/shoreline-land-water).

The same prompt, before any update, almost never says land. On 200 balanced places it scored 47 percent, with 5 percent of the land cells correct and every water cell correct. After training, a fresh 200 balanced places scored 75.5 percent: 61 percent of land cells and 92 percent of water cells. On 200 places drawn uniformly from the globe, the score was 81.5 percent: 72 percent on land and 86 percent on water.

That is a coarse map, not a coastline survey. The model has a better chance on open ocean and on broad interiors than on a cell that sits on the 1:110m shore.

## Try it

The map above takes a latitude and a longitude, or a click. It draws the atlas tile and shows the word the adapter gave for the one-degree cell that contains that place, next to the atlas label.

The code is at [github.com/saqlain2204/shoreline](https://github.com/saqlain2204/shoreline). A local check of one place:

```bash
python -m shoreline.demo --lat 48.86 --lon 2.35 --answer land
```
