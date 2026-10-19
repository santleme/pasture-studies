from dataclasses import replace

from farm_sim.memory import retrieve
from farm_sim.model import Position, ResourcePatch
from farm_sim.perception import perceive
from farm_sim.simulation import make_world


def test_nearest_resource_distance_ignores_depleted_patches():
    world = make_world(123)
    animal = world.animals[0]
    # The nearest forage is depleted; the live target is farther away. A
    # target-distance feature must describe the reachable/live target, not the
    # stale visual patch.
    depleted_forage = Position(animal.position.x, animal.position.y)
    available_forage = Position(min(world.width - 1, animal.position.x + 3), animal.position.y)
    depleted_water = Position(animal.position.x, animal.position.y)
    available_water = Position(animal.position.x, max(0, animal.position.y - 2))
    world = replace(world,
        forage=(ResourcePatch(depleted_forage, False), ResourcePatch(available_forage, True)),
        water=(ResourcePatch(depleted_water, False), ResourcePatch(available_water, True)))
    observed = perceive(world, animal, mode="path-aware-v2")
    assert observed.nearest_forage_distance == 3
    assert observed.nearest_water_distance == 2


def test_no_available_target_uses_out_of_vision_sentinel_distance():
    world = make_world(456)
    animal = world.animals[0]
    world = replace(world,
        forage=tuple(ResourcePatch(p.position, False) for p in world.forage),
        water=tuple(ResourcePatch(p.position, False) for p in world.water))
    observed = perceive(world, animal, mode="path-aware-v2")
    assert observed.nearest_forage_distance == 5
    assert observed.nearest_water_distance == 5


def test_path_aware_distance_uses_grid_detour_not_manhattan():
    world = make_world(777)
    animal = replace(world.animals[0], position=Position(2, 2))
    target = Position(4, 2)
    world = replace(world, animals=(animal, *world.animals[1:]),
                    obstacles=frozenset({Position(3, 2)}),
                    forage=(ResourcePatch(target, True),), water=())
    observed = perceive(world, animal, mode="path-aware-v2")
    assert observed.nearest_forage_distance == 4


def test_legacy_perception_mode_remains_reproducible():
    world = make_world(123)
    animal = world.animals[0]
    close_depleted = Position(animal.position.x, animal.position.y)
    far_available = Position(animal.position.x + 3, animal.position.y)
    world = replace(world, forage=(ResourcePatch(close_depleted, False), ResourcePatch(far_available, True)))
    assert perceive(world, animal, mode="legacy-v1").nearest_forage_distance == 0


def test_policy_loading_preserves_versioned_perception_mode(tmp_path):
    from farm_sim.policy import CategoricalPolicy, load_checkpoint, save_checkpoint
    checkpoint = CategoricalPolicy.initialize(321)
    checkpoint.perception_mode = "path-aware-v2"
    path = tmp_path / "candidate.npz"
    save_checkpoint(checkpoint, path, metadata={"config": {"perception_mode": "path-aware-v2"}})
    loaded = load_checkpoint(path)
    assert loaded.perception_mode == "path-aware-v2"
