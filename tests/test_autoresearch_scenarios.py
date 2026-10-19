from farm_sim.simulation import make_resource_rich_world, make_world


def test_resource_rich_profile_is_deterministic_and_does_not_mutate_default():
    original = make_world(91)
    first = make_resource_rich_world(91)
    second = make_resource_rich_world(91)
    assert original == make_world(91)
    assert first == second
    assert (len(first.forage), len(first.water)) == (8, 5)
    assert (len(original.forage), len(original.water)) == (3, 1)
    assert not (set(p.position for p in first.forage + first.water) & set(first.obstacles))


def test_resource_rich_profile_rejects_impossible_counts():
    import pytest
    with pytest.raises(ValueError, match="at least"):
        make_resource_rich_world(92, forage_count=2)
