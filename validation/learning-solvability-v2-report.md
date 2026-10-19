# Local solvability and collapse-cause experiments (v2 prototype)

## Outcome
The current default world is **not solvable as a sustainable 72-tick resource-maintenance task**: it contains exactly three forage patches and one water patch, each permanently consumed on success, while five animals take 72 decisions each and hunger/thirst increase every tick. This is a structural supply bound, not an optimizer defect. The existing v1 one-step classification task is learnable (historical report records 97% vs 14% accuracy), but that does not establish multi-tick control.

A no-training deterministic oracle uses only the existing 20 input features and action interface. On five paired starter seeds and 72 ticks/seed, the default non-renewing world yields exactly 4 total successful resource events per seed (20 total). A sandboxed v2 transition wrapper that regenerates a consumed patch after three ticks produces 36–55 successes/seed (216 total; 10.8×), without changing the reward function or counting reward as a success metric. The final metric is successful forage/water event count; the controller also considers observable hunger/thirst/fatigue. This demonstrates that a simple controller can repeatedly service resources when the environment supplies them. It does not demonstrate learned-policy performance.

A reward micro-probe independently shows that legacy `HERD_INTERACT` with social need already zero receives +0.0302 after drive costs despite no marginal social improvement. `validation.viability_v2.marginal_contact_reward` removes only the +0.1 contact bonus in that no-progress case, yielding −0.0698; when need falls, it preserves the contact bonus. Unit tests cover both cases. This is a measured incentive bug that can sustain HERD_INTERACT; it does not alone prove the entire 648k-action collapse was caused by it.

## Ranked explanations
1. **Demonstrated structural impossibility for sustained v1 needs — high confidence.** `farm_sim/simulation.py:41-59,180-183` creates 3 forage + 1 water and permanently marks successful patches unavailable; `farm_sim/physiology.py:19-22` increases hunger/thirst on every interval. At most four consumptions serve 360 animal-tick decisions. This can force unmet drives in any policy. It is the leading explanation for long-run `WALK`/resource-seeking failure, though it does not explain all policy preference by itself.
2. **Demonstrated reward exploit / immediate-objective mismatch — high confidence as defect, medium as sole collapse cause.** `farm_sim/physiology.py:44-50` pays a repeated nearby-herd bonus even after social need is zero. `validation/train_eval_mvp1.py:72-92,105-119,125-188` trains one-step action reward labels from random-collector states or four hand-written cases, not outcomes from the policy's own sustainable rollouts. The four-case labels explicitly map both negative-memory states to WALK (`:60-69`). Historical rollouts then show either >99% WALK or 100% HERD_INTERACT, depending on support (`MVP1_REMEDIATION_REPORT.md:25,56,60-68`).
3. **Transition/observation aliasing and inference mismatch — plausible contributor, not yet causal.** `farm_sim/simulation.py:151-178` gives all five agents the same pre-transition snapshot, then consumes resources and resolves occupancy in stable ID order. `farm_sim/model.py:104-110` and `farm_sim/perception.py:41-60` encode limited local features rather than relative herd/resource state; `farm_sim/policy.py:59-65` evaluates by argmax while training samples actions. These can make one-step labels fail under rollout, especially with shared scarce resources.

## Correction prototype and regression
No historical files, frozen thresholds, preregistration, checkpoints, or runtime modules were changed. The prototype lives separately in `validation/viability_v2.py`; it wraps the legacy simulation to supply renewable resources and has a narrow marginal-contact reward transform. The old fixed-resource/reward behavior remains byte-identical.

Reproduce:
```sh
.venv/bin/python -m pytest -q tests/test_learning_solvability_v2.py
.venv/bin/python -m pytest -q
```
Recorded verification after prototype: targeted viability tests **3 passed**; complete suite **201 passed, 2 skipped, 162 subtests passed** (72.40s). These tests prove the supply-bound claim, resource-renewal behavior improvement under a transparent oracle, and the no-progress herd-bonus correction. They do not prove an RL algorithm learns the improved behavior.

## Next discriminating studies (now that feasibility is shown)
- Run a small local, non-confirmatory learning pilot only on the renewable v2 scenario; compare same-seed REINFORCE/A2C, sampled-vs-argmax evaluation, and 1-step vs full-rollout training. Use successful resource events, time-to-resource, final hunger/thirst, repeated-action share, and navigation success as primary behavior metrics; reward is diagnostic only.
- Permute animal decision order under common random numbers. Large changes in resource success identify sequential arbitration/observability as a causal contributor.
- Preserve the 20-feature observation but compare (a) current reward, (b) marginal reward correction, and (c) recurrent or explicit resource-state features, one factor at a time. No expensive or Kaggle run is justified until the local pilot separates these effects.

## Boundary
These are synthetic software task variables only. The resource-renewal interval and reward transformation are research prototypes, not cattle biology or husbandry guidance. The current evidence establishes a viable controlled toy task and two corrected mechanics, **not** a validated learned agent or resolution of the locked MVP1 research gates.
