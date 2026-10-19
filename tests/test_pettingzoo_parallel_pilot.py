from dataclasses import replace

import numpy as np
import pytest
import importlib.util

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pettingzoo") is None or importlib.util.find_spec("gymnasium") is None,
    reason="optional PettingZoo/Gymnasium dependencies are not installed",
)

from farm_sim.model import Action, Position, ResourcePatch
from farm_sim.simulation import make_world


def _pilot_api():
    from validation.pettingzoo_parallel_pilot import ACTION_LIST, AGENT_IDS, PastureParallelEnv
    return ACTION_LIST, AGENT_IDS, PastureParallelEnv


def test_parallel_api_smoke_and_fixed_seed_reset_are_reproducible():
    _, AGENT_IDS, PastureParallelEnv = _pilot_api()
    from pettingzoo.test import parallel_api_test

    env = PastureParallelEnv(seed=23, max_cycles=8)
    first, _ = env.reset(seed=23)
    second, infos = env.reset(seed=23)
    assert first.keys() == second.keys() == set(AGENT_IDS)
    assert all(np.array_equal(first[agent], second[agent]) for agent in AGENT_IDS)
    assert all(infos[agent]["seed"] == 23 for agent in AGENT_IDS)
    assert all(env.observation_space(agent).contains(second[agent]) for agent in AGENT_IDS)
    assert all(env.action_space(agent).contains(0) for agent in AGENT_IDS)
    parallel_api_test(env, num_cycles=10)


def test_actions_are_collected_from_one_snapshot_then_resources_commit_in_id_order():
    ACTION_LIST, AGENT_IDS, PastureParallelEnv = _pilot_api()
    env = PastureParallelEnv(seed=2, max_cycles=4)
    env.reset(seed=2)
    world = make_world(2)
    spots = (Position(2, 2), Position(2, 3), Position(5, 5), Position(6, 5), Position(6, 4))
    animals = tuple(replace(animal, position=spot) for animal, spot in zip(world.animals, spots))
    env.sim.world = replace(
        world,
        animals=animals,
        forage=(ResourcePatch(Position(2, 2)),),
    )
    before_features = dict(env.sim.decision_features())
    collected = {}
    original_step = env.sim.step_with_decisions

    def capture_decisions(decisions):
        collected.update(decisions)
        return original_step(decisions)

    env.sim.step_with_decisions = capture_decisions
    observations, rewards, terminations, truncations, infos = env.step(
        {agent: ACTION_LIST.index(Action.GRAZE) for agent in AGENT_IDS}
    )
    after = {animal.animal_id: animal for animal in env.sim.world.animals}

    assert env.sim.world.tick == 1
    assert infos["cow-1"]["observation_snapshot_tick"] == infos["cow-5"]["observation_snapshot_tick"] == 0
    assert infos["cow-1"]["commit_order"] == list(AGENT_IDS)
    assert before_features.keys() == observations.keys()
    assert {agent: decision[1] for agent, decision in collected.items()} == before_features
    assert not any(terminations.values()) and not any(truncations.values())
    assert rewards["cow-1"] > rewards["cow-2"]
    assert after["cow-1"].last_reward > after["cow-2"].last_reward
    assert env.sim.world.forage[0].available is False
    assert not any(event.kind == "forage_success" and event.animal_id == "cow-2" for event in env.sim.events)
    assert all(np.isfinite(reward) for reward in rewards.values())


def test_invalid_or_incomplete_action_batch_does_not_advance_simulation():
    ACTION_LIST, AGENT_IDS, PastureParallelEnv = _pilot_api()
    env = PastureParallelEnv(seed=5)
    before = env.sim.world
    with pytest.raises(ValueError, match="exactly one action"):
        env.step({"cow-1": 0})
    with pytest.raises(ValueError, match="out of range"):
        env.step({agent: (len(ACTION_LIST) if agent == "cow-1" else 0) for agent in AGENT_IDS})
    assert env.sim.world == before
