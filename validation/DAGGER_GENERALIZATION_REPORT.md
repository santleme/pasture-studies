# Pasture Studies: DAgger generalization study

## Decision

**Bounded, five-seed evidence does not clear the preregistered robust-win gate.**
Domain-randomized DAgger is a useful candidate, not a proven replacement: it
helps on unseen layouts and on the combined unseen profile, but its combined
behavior advantage over fixed-domain DAgger is inconsistent across training
seeds. The final validation therefore records `demonstrated_blocker`, not
successful generalization. Keep the current policy as an experiment; do not
promote a checkpoint or claim that the generalization problem is solved.

## Setup

`paired-instrumented-five-seeds.json` compares fixed-domain and domain-randomized
DAgger with the same shared per-animal MLP, 65-value history/domain observation,
60 initial teacher episodes, and 8 DAgger rounds × 30 episodes. Five training
seeds (1–5) each use the same five held-out reset seeds (301, 307, 313, 317,
331); episodes are measured in six domains. Training profiles are nominal
8/6/regrowth-1 plus 9/7/rate-3, 10/8/rate-3, and randomized-layout 12/10/rate-3.
Thus the changed layout, 9/7 supply, and regrowth-2 tests remain outside the
training-profile set. Metrics below average animals and evaluation seeds unless
identified as worst-case; need is lower-is-better. Reward is not used as a
success criterion.

## Results (five-seed means)

| Held-out domain | Policy | Forage events | Water events | Hunger | Thirst | Teacher action disagreement |
|---|---|---:|---:|---:|---:|---:|
| Random layout (10/8) | Fixed DAgger | 9.936 | 9.376 | 0.155 | 0.153 | 4.5% |
|  | Domain-randomized | **10.088** | **9.592** | **0.135** | **0.141** | **2.8%** |
| Regrowth 2 (8/6) | Fixed DAgger | 9.752 | 8.952 | 0.159 | 0.159 | 3.8% |
|  | Domain-randomized | **9.816** | **8.984** | **0.150** | 0.162 | **2.5%** |
| Combined unseen (random layout, 12/10) | Fixed DAgger | 10.288 | 10.328 | 0.110 | 0.092 | 5.6% |
|  | Domain-randomized | **10.384** | 10.328 | **0.098** | **0.091** | **2.1%** |

On the combined domain, directional wins in the four independent measures
(forage↑, water↑, hunger↓, thirst↓) were 2/4, 2/4, 4/4, 2/4, and 4/4 for
training seeds 1–5. Only **2/5** seeds beat the fixed reference on at least
three of four measures, below the required 4/5. Pooled behavior improves in
forage (+0.096 events), hunger (−0.013), and thirst (−0.001), while water
events are tied (10.328 each). This is a real but partial average gain, not a
robust per-seed result.

Across every domain and training seed, the candidate's worst per-animal mean
need was 0.285, action share never exceeded 0.50 (the 0.90 cap), and the minimum
per-animal forage+water event count relative to the transparent teacher was
0.947 (the floor is 0.80). Thus failure is not explained by the behavioral
floor, a collapsed action, or obvious lack of teacher-solvability.

## Causal localization

1. **Most imitation errors are low-need action choices, not missed urgent
   resources.** In the randomized candidate over five runs × five evaluation
   seeds, disagreement events classified as high-need were 15/9,000 in layout
   shift and 15/9,000 in regrowth shift; combined unseen had 0/9,000. Other
   action disagreements were 233, 208, and 193 respectively. This points to
   teacher/student action timing and choice calibration, rather than a frequent
   emergency underfeeding mode.
2. **Resource arbitration is not the dominant observed failure.** A new
   counterfactual preview mirrors `Simulation`'s stable animal-ID transition
   order and resource reservation, instead of counting any consumed patch as a
   conflict. Of 6,880 graze/drink requests on regrowth-shift, 2,350 succeeded
   in that tick and 4,530 moved toward a target; none were classified as a
   same-tick contention or a no-target failure. The preview's predicted
   successes match actual simulator commits in a regression test.
3. **Due-regrowth observability matters modestly.** Exposing patches whose
   regrowth deadline is already due (without changing transitions) on the
   regrowth-2 domain yielded 9.816/8.984 forage/water events and
   0.150/0.162 hunger/thirst, versus the deliberately stale observation's
   9.776/8.936 and 0.159/0.177. This identifies a real timing limitation, but
   the size of the effect alone cannot account for the inconsistent combined
   result.
4. **History and domain features affect action choice, but are not sufficient
   fixes.** Paired masking of prior-observation features raised teacher
   disagreement by 7.5 percentage points on regrowth shift; masking domain
   context raised it by 6.3 points. However, the resulting need metrics did not
   uniformly worsen. These features causally influence decisions, but current
   evidence does not establish that more history/context alone yields better
   behavior.
5. **Capacity stress is a separate boundary.** Earlier stress screens (kept as
   `screen-v0-stress-infeasible-*`) found settings where the transparent teacher
   itself fails. They are not used as policy generalization failures or as
   training targets. The final held-out profiles remain teacher-solvable.

## Approaches tried and next proposal

- Fixed vs. randomized-domain DAgger: randomization improved randomized-layout
  and combined-suite averages and reduced imitation disagreement, but missed
  seed consistency.
- Richer history/domain observation and paired masking: history/context changes
  decisions but does not reliably improve need outcomes.
- Wider and joint model screens: no repeatable multi-seed behavioral win was
  established; keep the simple shared per-animal MLP.
- PPO fine-tuning was evaluated as a **one-animal** comparison on five training
  seeds and suggests useful single-agent potential, but that result does not
  validate five-animal generalization. Do not transfer it to the five-cow claim.

Next, retain DAgger's transparent teacher as the safe baseline; prioritize
action-label calibration around low-need states, recurrent/resource-deadline
features with ablations on training domains, and then re-run a locked combined
test across at least five fresh training seeds. Keep domain randomization as a
candidate curriculum, not an established winner. A multi-agent RL fine-tune is
not justified until its training and evaluation use the same behavioral gates.

## Reproducibility and validation

- Main paired artifact: `.omx/specs/autoresearch-dagger-generalization/paired-instrumented-five-seeds.json`
- Earlier exploratory screens and the one-animal PPO comparison remain in the
  same spec folder; they are not substituted for the final five-animal test.
- Core simulator files under `farm_sim/` were not changed.
- Tests: `232 passed, 3 skipped, 162 subtests passed`; the existing SB3 shape
  warning is non-failing.
