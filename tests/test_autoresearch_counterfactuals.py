from farm_sim.policy import CategoricalPolicy
from validation.compare_autoresearch_checkpoints import _paired_cognition_rollouts


def test_paired_cognition_counterfactuals_change_only_declared_inputs():
    policy = CategoricalPolicy.initialize(101)
    policy.perception_mode = "path-aware-v2"
    result = _paired_cognition_rollouts(policy)
    assert result["fixture_seed"] == 83831
    for factor in ("memory", "perception"):
        comparison = result[factor]
        assert comparison["base"]["tick"] == comparison["changed"]["tick"] == 73
        assert set(comparison["base"]["final_drives"]) == {f"cow-{i}" for i in range(1, 6)}
        assert "reward_delta" in comparison
        assert comparison["base"]["action_counts"] != {}  # all five animals were simulated
    assert "water_access success -> blocked" in result["memory"]["changed_only"]
    assert "world unchanged" in result["perception"]["changed_only"]
