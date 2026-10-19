# Consolidated RL stack pilot: Gymnasium + Stable-Baselines3

**Decision:** the optional adapter and PPO training loop work at one- and five-animal scales, and PPO beats the seeded random policy on this task's synthetic episodic return. This is an integration/learnability signal, **not** evidence that learned behavior is useful against the existing need-controller or that the task is sustainable. Do not promote a checkpoint or start a larger/Kaggle run yet.

## Scope

- Reused `Simulation.step_with_decisions`; no simulator or physiology rewrite.
- Kept Gymnasium/SB3 as optional dependencies (`requirements-rl.txt`); the shipped NumPy runtime has no new dependency.
- One-animal mode controls cow-1 while four background animals REST. Five-animal mode uses one shared PPO policy over a joint `MultiDiscrete` action; simulator commits actions/resources in its existing stable-ID order.
- This is a synthetic toy task, not animal or husbandry validation.

## PPO vs seeded random (screening run)

Command (local CPU, seed 11, 10,240 requested training steps per mode, 20 held-out episodes):

```sh
.venv/bin/python -m scripts.train_sb3_ppo \
  --output /tmp/pasture-sb3-eval --seed 11 \
  --timesteps 10240 --evaluation-episodes 20
```

| Mode | PPO mean episode return | Random mean episode return | Difference |
|---|---:|---:|---:|
| One controlled animal | 2.212 | -1.730 | +3.942 |
| Five-animal joint | 2.101 | -1.659 | +3.760 |

Evaluation uses the same held-out reset seeds (10011–10030) for PPO and random. The full episode returns and per-animal returns are written to the command's `comparison-report.json`; checkpoints are written alongside it. One training seed and return-only comparison are exploratory, not a robust efficacy gate.

## Behavior gate against the existing heuristic

The separate, more behavior-focused validation wrapper (`validation/sb3_pilot.py`) regrows forage and water patches after three ticks and uses four `NeedController` context animals for the single-cow trial. It measures resource successes and physiology independently of reward. Reproduction:

```sh
.venv/bin/python -m validation.sb3_pilot --agents 1 --steps 7200 --seed 1
.venv/bin/python -m validation.sb3_pilot --agents 5 --steps 3600 --seed 1
```

The one-cow PPO did **not** match the heuristic: 50.7 vs 53.0 resource successes per episode, controlled-cow final hunger 1.00 vs 0.094, and thirst 1.00 for both. In the five-agent task PPO yielded 12.0 resource successes and mean world hunger 0.807, compared with 41.7 and 0.490 for the heuristic. Thirst also reached 1.00 even for the heuristic, exposing a remaining supply/viability defect. These results are a screening failure for useful behavior, despite the better return than random in the other, REST-background task; the task setups and metrics differ and should not be conflated.

The PettingZoo Parallel prototype exposes all five actions from the same pre-transition snapshot but delegates to sequential stable-ID commits; it is an interface/API pilot, not a simultaneous-physics claim or a trained multi-agent result. TorchRL was not adopted: SB3 already supplies the conventional PPO baseline and there is no evidence that another training stack would resolve task viability or the behavior gap.

## Recommendation

Keep this optional pilot, but do not promote its checkpoint or expand compute. Before a larger cohort, define and test a sustainable scenario with adequate forage **and water**, and freeze a behavior-only per-animal success gate. Then repeat the one-animal comparison across multiple training seeds against both random and the need-controller. Only if that gate passes should the five-animal follow-up compare the shared-policy joint-action baseline with a PettingZoo-compatible independent/shared-policy method, while preserving ordered resource arbitration in the declared environment contract. Keep held-out worlds and all locked MVP1 evidence separate.
