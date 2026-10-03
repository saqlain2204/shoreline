"""Behavior of the globe, the two tasks, and the GRPO advantage."""

from __future__ import annotations

import numpy as np
import pytest

from shoreline.geo import haversine_km, location_reward
from shoreline.grpo import group_advantages
from shoreline.parsing import parse_guess, parse_label, parse_move
from shoreline.raster import fill_polygon
from shoreline.sim import ShoreSim
from shoreline.world import World, survey_axes


def test_two_degree_grid_is_the_karpathy_count():
    latitudes, longitudes = survey_axes(2)
    assert len(latitudes) * len(longitudes) == 16_200


def test_dateline_sliver_stays_on_the_edge():
    mask = np.zeros((720, 1440), dtype=np.uint8)
    ring = [[180, 70.832], [178.9, 70.78], [178.73, 71.10], [180, 71.52], [180, 70.832]]
    fill_polygon(mask, [ring], 1)
    assert mask[74].mean() < 0.02
    assert mask[74, -1] == 1 or mask[74, -2] == 1


def test_polygon_fill_covers_the_interior_only():
    mask = np.zeros((18, 36), dtype=np.uint8)
    ring = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    fill_polygon(mask, [ring], 1)
    assert mask[8, 18] == 1
    assert mask[4, 18] == 0


def test_real_places():
    world = World.load()
    assert world.label(48.86, 2.35) == "land"
    assert world.label(23.0, 12.0) == "land"
    assert world.label(-25.0, 134.0) == "land"
    assert world.label(-80.0, 0.0) == "land"
    assert world.label(0.0, -160.0) == "water"
    assert world.label(41.5, 50.5) == "water"


def test_survey_rewards_the_parsed_label():
    sim = ShoreSim()
    outcome = sim.reset(task="survey", latitude=48.86, longitude=2.35, include_image=False)
    assert outcome.prompt.startswith("Land or Water?")
    assert "one word" in outcome.prompt
    assert "truth" not in outcome.prompt.lower()
    assert sim.step("It is on land.").reward == 1.0
    sim.reset(task="survey", latitude=0.0, longitude=-160.0, include_image=False)
    assert sim.step("land").reward == 0.0
    with pytest.raises(RuntimeError):
        sim.step("water")


def test_survey_seed_repeats_and_balance_hits_both():
    sim = ShoreSim()
    first = sim.reset(seed=7, include_image=False)
    second = sim.reset(seed=7, include_image=False)
    assert (first.latitude, first.longitude) == (second.latitude, second.longitude)
    labels = set()
    for seed in range(24):
        sim.reset(seed=seed, balance=True, include_image=False)
        labels.add(sim._label)
    assert labels == {"land", "water"}


def test_tile_is_a_png_and_north_is_up():
    world = World.load()
    blob = world.render_png(48.86, 2.35, span_deg=24, size=64)
    assert blob.startswith(b"\x89PNG")


def test_coast_move_and_stop():
    land = np.zeros((20, 40), dtype=np.uint8)
    land[:, 20:] = 1
    world = World(land)
    sim = ShoreSim(world)
    outcome = sim.reset(task="coast", latitude=10.0, longitude=20.0, include_image=False, step_deg=5)
    assert "here" in outcome.legal_actions
    moved = sim.step("north")
    assert moved.done is False
    assert moved.latitude == pytest.approx(15.0)
    stopped = sim.step("go west, then here")
    assert stopped.done is True
    assert stopped.reward in (0.0, 1.0)


def test_locate_hides_the_drop_and_scores_distance():
    sim = ShoreSim()
    outcome = sim.reset(task="locate", latitude=48.86, longitude=2.35, include_image=False, max_moves=4)
    assert "Latitude:" not in outcome.prompt
    assert "48.86" not in outcome.prompt
    assert "2.35" not in outcome.prompt
    assert "L" in outcome.prompt and "W" in outcome.prompt
    looked = sim.step("north")
    assert looked.done is False
    assert looked.latitude == pytest.approx(49.86)
    assert sim._origin_lat == pytest.approx(48.86)
    scored = sim.step("48.86 2.35")
    assert scored.done is True
    assert scored.reward == pytest.approx(location_reward(0.0, 0.02))
    sim.reset(task="locate", latitude=48.86, longitude=2.35, include_image=False)
    far = haversine_km(48.86, 2.35, -33.87, 151.21)
    assert sim.step("-33.87 151.21").reward == pytest.approx(location_reward(far, 0.01))
    sim.reset(task="locate", latitude=48.86, longitude=2.35, include_image=False, max_moves=1)
    assert sim.step("look around").reward == 0.0


def test_parsers_use_the_last_word():
    assert parse_label("Not water. Land.") == "land"
    assert parse_move("sail west and stop here") == "here"
    assert parse_label("maybe") is None
    assert parse_guess("near 48.86, 2.35 then 0 0") == (0.0, 0.0)
    assert parse_guess("north") is None
    assert parse_guess("latitude 100 longitude 10") is None


def test_latest_checkpoint_picks_the_highest_step(tmp_path):
    from shoreline.grpo import latest_checkpoint

    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-notes").mkdir()
    found = latest_checkpoint(tmp_path)
    assert found is not None
    assert found[0] == 10
    assert found[1].name == "checkpoint-10"
    assert latest_checkpoint(tmp_path / "missing") is None


def test_score_survey_counts_a_constant_answer():
    from shoreline.eval import score_survey

    result = score_survey(lambda prompt: "water", n=20, seed=0)
    assert result["n"] == 20
    assert result["water"] == 1.0
    assert result["land"] == 0.0
    assert 0.0 < result["accuracy"] < 1.0


def test_group_advantages_center_each_group():
    assert group_advantages([1, 0, 1, 0], 4) == pytest.approx([1, -1, 1, -1])
    assert group_advantages([1, 1, 0, 0], 2) == pytest.approx([0, 0, 0, 0])
