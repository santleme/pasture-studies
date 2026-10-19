#!/usr/bin/env python3
"""Frozen-seed NumPy training and rollout evaluation for MVP 1 remediation.

All rewards/physiology/scenarios are synthetic software conventions. This script
never reads locked test outcomes until it has completed training, validation
selection, and frozen analysis configuration; it writes only to its --output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from farm_sim.execution_policy import require_kaggle_for_full_workload
from farm_sim.model import Action, FEATURE_NAMES, ToyPhysiology, Position, ResourcePatch
from farm_sim.policy import CategoricalPolicy, HeuristicBaseline, save_checkpoint, load_checkpoint
from farm_sim.simulation import Simulation, make_world
from farm_sim.perception import perceive, encode_features
from farm_sim.memory import retrieve
from farm_sim.navigation import navigate
from farm_sim.physiology import step_physiology

MANIFEST = ROOT / ".omx/research/mvp1-remediation-baseline/experiment-manifest.json"
LR = 0.07
BATCH = 4
CHECKPOINT_EPISODES = (500, 1000, 1500, 2000)
DAYS = 3
TICKS = 72
BOOTSTRAPS = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text())


def softmax_logits(policy: CategoricalPolicy, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(x @ policy.w1 + policy.b1)
    logits = hidden @ policy.w2 + policy.b2
    probs = np.exp(logits - np.max(logits)); probs /= probs.sum()
    return hidden, probs


def task_cases() -> tuple[tuple[np.ndarray, Action, str], ...]:
    names = {name: i for i, name in enumerate(FEATURE_NAMES)}
    water = np.zeros(len(FEATURE_NAMES)); water[[names[k] for k in ("thirst", "water_visible", "water_available", "mem_water_success")]] = [0.9, 1, 1, 1]
    water[names["water_distance_norm"]] = .1
    water_blocked = water.copy(); water_blocked[names["mem_water_success"]] = 0
    forage = np.zeros(len(FEATURE_NAMES)); forage[[names[k] for k in ("hunger", "forage_visible", "forage_available", "mem_forage_success")]] = [.2, 1, 1, 1]
    forage[names["forage_distance_norm"]] = .1
    forage_depleted = forage.copy(); forage_depleted[names["mem_forage_success"]] = 0
    return ((water, Action.DRINK, "water-memory-positive"), (water_blocked, Action.WALK, "water-memory-negative"),
            (forage, Action.GRAZE, "forage-memory-positive"), (forage_depleted, Action.WALK, "forage-memory-negative"))


def one_step_rewards(world, animal, features: np.ndarray) -> np.ndarray:
    """Exact one-tick current simulation reward for each possible action at a state."""
    output = []
    for action in Action:
        occupied = {a.position for a in world.animals if a.animal_id != animal.animal_id}
        nav = navigate(world, animal, action, occupied)
        fp = next((p for p in world.forage if p.position == nav.position), None)
        wp = next((p for p in world.water if p.position == nav.position), None)
        nearby = any(abs(nav.position.x-p.x)+abs(nav.position.y-p.y) <= 2 for p in occupied)
        _, reward, _ = step_physiology(animal.physiology, action,
            forage_available=(fp.available if fp else None), water_available=(wp.available if wp else None),
            herd_nearby=nearby)
        output.append(reward)
    # The confirmatory encoder condition changes only its input support. Keep
    # the preregistered +1 / -.25 schedule shared with the four-case condition;
    # for an encoder state, "correct" is an action attaining the highest
    # unchanged one-step toy physiology reward in that state. No diversity term
    # or reward shaping is introduced.
    scores = np.asarray(output, dtype=np.float64)
    optimal = scores == np.max(scores)
    return np.where(optimal, 1.0, -0.25)


class RandomCollector:
    """Seeded behavior policy used only to sample training-only state support."""
    def __init__(self, rng: np.random.Generator): self.rng = rng
    def decide(self, features):
        probs = np.full(len(Action), 1 / len(Action))
        action = tuple(Action)[int(self.rng.choice(len(Action), p=probs))]
        from farm_sim.model import PolicyDecision
        return PolicyDecision(action, tuple(probs), "training-state-collector")


def collect_encoder_states(seeds: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate immutable feature/reward samples only from preregistered train worlds."""
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    # Separate deterministic collection stream from world generation and policy-training RNG.
    rng = np.random.default_rng(730_000 + len(seeds))
    for world_seed in seeds:
        sim = Simulation(make_world(world_seed), RandomCollector(rng))
        for _ in range(TICKS):
            before = sim.world
            for animal in before.animals:
                observation = perceive(before, animal)
                memory = retrieve(animal.memory, before.tick)
                x = np.asarray(encode_features(observation, memory, animal, before.width, before.height), dtype=np.float64)
                rows.append((x, one_step_rewards(before, animal, x)))
            sim.step()
    if not rows or any(not np.isfinite(x).all() or not np.isfinite(r).all() for x, r in rows):
        raise ValueError("encoder training data is empty or non-finite")
    return rows


def update_batch(policy: CategoricalPolicy, examples: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator) -> float:
    """REINFORCE with exact action-independent expected-reward baseline per sampled state."""
    gw1 = np.zeros_like(policy.w1); gb1 = np.zeros_like(policy.b1)
    gw2 = np.zeros_like(policy.w2); gb2 = np.zeros_like(policy.b2)
    returns = []
    for x, rewards in examples:
        hidden, probs = softmax_logits(policy, x)
        chosen = int(rng.choice(len(Action), p=probs))
        advantage = rewards[chosen] - float(probs @ rewards)
        dlogits = -probs.copy(); dlogits[chosen] += 1
        dlogits *= advantage
        gw2 += np.outer(hidden, dlogits); gb2 += dlogits
        dz = (policy.w2 @ dlogits) * (1 - hidden * hidden)
        gw1 += np.outer(x, dz); gb1 += dz
        returns.append(float(rewards[chosen]))
    grads = [g * (LR / len(examples)) for g in (gw1, gb1, gw2, gb2)]
    norm = math.sqrt(sum(float(np.sum(g*g)) for g in grads))
    if not math.isfinite(norm): raise FloatingPointError("non-finite training gradient")
    clip = min(1, 2 / max(norm, 1e-12))
    policy.w1 += grads[0]*clip; policy.b1 += grads[1]*clip
    policy.w2 += grads[2]*clip; policy.b2 += grads[3]*clip
    if not np.isfinite(policy.parameter_vector()).all(): raise FloatingPointError("non-finite weights")
    return float(np.mean(returns))


def train_four_case(seed: int, output: Path, condition: str) -> tuple[dict[int, CategoricalPolicy], CategoricalPolicy, dict[str, Any]]:
    policy = CategoricalPolicy.initialize(seed); initial = policy.copy()
    policy.policy_id = f"{condition}-seed-{seed}"
    rng = np.random.default_rng(seed + 1_000_003)
    cases = task_cases(); trace = []; candidates = {}; curves = []
    for episode in range(1, 2001):
        order = rng.permutation(4)
        for ix in order:
            x, target, _ = cases[int(ix)]
            hidden, probs = softmax_logits(policy, x)
            chosen = int(rng.choice(len(Action), p=probs)); reward = 1.0 if chosen == tuple(Action).index(target) else -.25
            advantage = reward - (1.0 * probs[tuple(Action).index(target)] - .25 * (1-probs[tuple(Action).index(target)]))
            dlogits = -probs.copy(); dlogits[chosen] += 1; dlogits *= advantage
            trace.append((x, hidden, dlogits))
        # Same balanced stochastic batch updates as approved existing trainer.
        gw1=np.zeros_like(policy.w1);gb1=np.zeros_like(policy.b1);gw2=np.zeros_like(policy.w2);gb2=np.zeros_like(policy.b2)
        batch=trace[-BATCH:]
        for x, hidden, dlogits in batch:
            gw2 += np.outer(hidden,dlogits);gb2 += dlogits
            dz=(policy.w2@dlogits)*(1-hidden*hidden);gw1 += np.outer(x,dz);gb1 += dz
        grads=[g*(LR/BATCH) for g in (gw1,gb1,gw2,gb2)]
        norm=math.sqrt(sum(float(np.sum(g*g)) for g in grads)); clip=min(1,2/max(norm,1e-12))
        policy.w1+=grads[0]*clip;policy.b1+=grads[1]*clip;policy.w2+=grads[2]*clip;policy.b2+=grads[3]*clip
        curves.append(float(np.mean([1 if int(np.argmax(policy.probabilities(cases[int(ix)][0])))==tuple(Action).index(cases[int(ix)][1]) else -.25 for ix in range(4)])))
        if episode in CHECKPOINT_EPISODES:
            candidate=policy.copy();candidate.episodes=episode; candidates[episode]=candidate
    return candidates, initial, {"episode_greedy_mean_reward":curves, "episodes":2000}


def train_encoder(seed: int, samples, output: Path, condition: str) -> tuple[dict[int, CategoricalPolicy], CategoricalPolicy, dict[str, Any]]:
    policy=CategoricalPolicy.initialize(seed); initial=policy.copy(); policy.policy_id=f"{condition}-seed-{seed}"
    rng=np.random.default_rng(seed+1_000_003); candidates={}; curves=[]
    for episode in range(1,2001):
        ix=rng.integers(0,len(samples),size=BATCH)
        batch=[samples[int(i)] for i in ix]
        curves.append(update_batch(policy,batch,rng))
        if episode in CHECKPOINT_EPISODES:
            candidate=policy.copy();candidate.episodes=episode;candidates[episode]=candidate
    return candidates,initial,{"batch_sampled_toy_reward":curves,"episodes":2000,"training_sample_count":len(samples)}


def make_scenario(family: str, seed: int):
    if family == "primary_starter_14x10": return make_world(seed)
    if family == "wide_grid_20x14": return make_world(seed,width=20,height=14)
    if family == "resource_scarce_relocated":
        world=make_world(seed)
        return replace(world, forage=(ResourcePatch(Position(1,8),True),ResourcePatch(Position(2,2),False),ResourcePatch(Position(3,7),False)),
                       water=(ResourcePatch(Position(12,8),True),))
    raise ValueError(family)


def valid(action, probs):
    return action in Action and len(probs)==len(Action) and all(math.isfinite(p) and 0<=p<=1 for p in probs) and abs(sum(probs)-1)<1e-6


def comparison_margins(baseline_mean: float, near_zero_absolute_margin: float) -> tuple[float, float, str]:
    if abs(baseline_mean) >= .1:
        return .05*abs(baseline_mean), .10*abs(baseline_mean), "baseline_absolute_mean"
    # The approved near-zero rule uses one frozen absolute fallback for both
    # utility and harm; do not double the harm threshold in this branch.
    return near_zero_absolute_margin, near_zero_absolute_margin, "frozen_pilot_absolute_fallback"


def resource_event_tally(events) -> Counter:
    kinds={"forage_success","forage_depleted","water_success","water_blocked"}
    counts=Counter()
    for event in events:
        if event.kind in kinds: counts[event.kind]+=1
    return counts


def rollout(policy, family: str, seed: int, detailed: bool = False) -> dict[str, Any]:
    sim=Simulation(make_scenario(family,seed),policy)
    actions=Counter();reasons=Counter();rewards=[];invalid=0;interactions=Counter();drive_trajectory=[];animal_states=[];resource_states=[]
    resource_event_kinds={"forage_success","forage_depleted","water_success","water_blocked"}
    def resource_state(world):
        return {"forage_available":sum(p.available for p in world.forage),
                "forage_depleted":sum(not p.available for p in world.forage),
                "water_available":sum(p.available for p in world.water),
                "water_depleted":sum(not p.available for p in world.water)}
    prior_resource_state=resource_state(sim.world)
    resource_states.append({"tick":sim.world.tick,**prior_resource_state})
    for _ in range(TICKS):
        result=sim.step(); tick_reward=0.0
        interactions.update(resource_event_tally(result.events))
        current_resource_state=resource_state(result.world)
        interactions["forage_patches_depleted_this_tick"] += max(0, current_resource_state["forage_depleted"]-prior_resource_state["forage_depleted"])
        interactions["water_patches_depleted_this_tick"] += max(0, current_resource_state["water_depleted"]-prior_resource_state["water_depleted"])
        prior_resource_state=current_resource_state
        resource_states.append({"tick":result.world.tick,**current_resource_state})
        for a in result.world.animals:
            actions[a.action.value]+=1;tick_reward+=a.last_reward
            if not valid(a.action,a.last_probabilities):invalid+=1
            for k in ("hunger","thirst","fatigue","social_need"): interactions[f"{k}_sum"]+=getattr(a.physiology,k)
            interactions[f"{a.action.value}_reward"]+=a.last_reward
            interactions[f"{a.action.value}_count"]+=1
        # Capture exclusive navigation classes independent of old aggregate compatibility reason.
        # Step internals are instrumented via deterministic replay of same pre-transition actions below.
        rewards.append(tick_reward)
        drive_trajectory.append({k:round(sum(getattr(a.physiology,k) for a in result.world.animals)/len(result.world.animals),6) for k in ("hunger","thirst","fatigue","social_need")})
        if detailed: animal_states.append([{"id":a.animal_id,"x":a.position.x,"y":a.position.y,"action":a.action.value,"reward":a.last_reward,
            "probabilities":list(a.last_probabilities),"features":list(a.last_features),"physiology":{"hunger":a.physiology.hunger,"thirst":a.physiology.thirst,"fatigue":a.physiology.fatigue,"social_need":a.physiology.social_need}} for a in result.world.animals])
    final={k:{"mean":float(np.mean([getattr(a.physiology,k) for a in sim.world.animals])),"min":min(getattr(a.physiology,k) for a in sim.world.animals),"max":max(getattr(a.physiology,k) for a in sim.world.animals)} for k in ("hunger","thirst","fatigue","social_need")}
    total=float(sum(rewards));counts={a.value:actions[a.value] for a in Action};n=TICKS*len(sim.world.animals)
    probs=np.asarray([counts[a.value]/n for a in Action]);entropy=float(-sum(p*math.log(p) for p in probs if p))
    result={"family":family,"world_seed":seed,"ticks":TICKS,"synthetic_days":DAYS,"return":total,"action_counts":counts,
       "action_fractions":{a.value:counts[a.value]/n for a in Action},"action_entropy_nats":entropy,"argmax_action_share":max(counts.values())/n,
       "invalid_outputs":invalid,"navigation_reason_counts":dict(reasons),"resource_interaction_and_action_reward_counts":dict(interactions),
       "resource_interactions":{"event_counts":{kind:interactions[kind] for kind in sorted(resource_event_kinds)},
           "patch_depletions":{"forage":interactions["forage_patches_depleted_this_tick"],"water":interactions["water_patches_depleted_this_tick"]},
           "final_resource_state":resource_states[-1]},
       "resource_state_trajectory_by_tick":resource_states,
       "reward_trajectory_per_tick_all_animals":[round(v,6) for v in rewards],"mean_reward_per_tick":total/TICKS,
       "drive_trajectory_mean_by_tick":drive_trajectory,"final_toy_physiology":final,"final_positions":{a.animal_id:[a.position.x,a.position.y] for a in sim.world.animals},
       "bounds_ok":all(0<=a.position.x<sim.world.width and 0<=a.position.y<sim.world.height for a in sim.world.animals)}
    if detailed: result["animal_state_trajectory_by_tick"]=animal_states
    return result


def get_reason_counts(policy, family: str, seed: int) -> Counter:
    """Replay with an observing wrapper to count the exclusive reason-coded nav classes."""
    import farm_sim.simulation as module
    original=module.navigate; counts=Counter()
    def wrapped(*args,**kwargs):
        value=original(*args,**kwargs);counts[value.subreason]+=1;return value
    module.navigate=wrapped
    try: rollout(policy,family,seed,False)
    finally: module.navigate=original
    return counts


def quick_return(policy, family: str, seed: int) -> float:
    return rollout(policy,family,seed,False)["return"]


def bootstrap_ci(differences: dict[int, dict[int, float]], n_boot: int, seed: int) -> dict[str, float]:
    """Crossed checkpoint/world bootstrap preserving the paired cell differences."""
    cp=sorted(differences); worlds=sorted(next(iter(differences.values())))
    vals=np.asarray([[differences[c][w] for w in worlds] for c in cp],dtype=float)
    rng=np.random.default_rng(seed); estimates=np.empty(n_boot)
    for i in range(n_boot):
        ci=rng.integers(0,len(cp),len(cp)); wi=rng.integers(0,len(worlds),len(worlds))
        estimates[i]=vals[np.ix_(ci,wi)].mean()
    # Two-way cluster-robust sensitivity (checkpoint ID × world ID), using the
    # Cameron-Gelbach-Miller inclusion/exclusion variance estimator.
    c,w=vals.shape; residual=vals-vals.mean(); n=c*w
    row_scores=residual.sum(axis=1); col_scores=residual.sum(axis=0)
    variance=(c/(c-1)*float(row_scores@row_scores)+w/(w-1)*float(col_scores@col_scores)
              -(n/(n-1))*float(np.sum(residual*residual)))/(n*n) if c>1 and w>1 else float("nan")
    cluster_se=math.sqrt(max(0.0,variance))
    return {"mean":float(vals.mean()),"ci95_low":float(np.quantile(estimates,.025)),"ci95_high":float(np.quantile(estimates,.975)),"bootstrap_replicates":n_boot,
            "training_seed_count":len(cp),"world_seed_count":len(worlds),
            "two_way_cluster_robust_sensitivity":{"standard_error":cluster_se,"ci95_low":float(vals.mean()-1.96*cluster_se),"ci95_high":float(vals.mean()+1.96*cluster_se),
              "method":"two-way clustered checkpoint and world seed, Cameron-Gelbach-Miller inclusion/exclusion"}}


def family_eval(condition, saved, initials, output: Path, manifest, pilot: dict[str, Any]) -> dict[str,Any]:
    seeds=manifest["seed_protocol"]["locked_test_world_seeds_by_family"]
    policy_classes={"trained":saved,"untrained":initials,"heuristic":None}
    full=[];differences={fam:{"vs_untrained":{},"vs_heuristic":{}} for fam in seeds}
    all_returns={}
    for family, world_seeds in seeds.items():
        for world_seed in world_seeds:
            heuristic=HeuristicBaseline()
            hrow=rollout(heuristic,family,world_seed,True)
            hrow.update({"condition":condition,"policy_group":"heuristic","training_seed":None,"checkpoint_episode":None})
            hrow["navigation_reason_counts"]=dict(get_reason_counts(HeuristicBaseline(),family,world_seed));full.append(hrow)
            for training_seed in sorted(saved):
                for group, collection in (("trained",saved),("untrained",initials)):
                    policy=collection[training_seed]["policy"] if group=="trained" else collection[training_seed]
                    row=rollout(policy,family,world_seed,True)
                    row.update({"condition":condition,"policy_group":group,"training_seed":training_seed,
                                "checkpoint_episode":saved[training_seed]["selected_episode"] if group=="trained" else 0})
                    row["navigation_reason_counts"]=dict(get_reason_counts(policy,family,world_seed));full.append(row)
                    all_returns[(family,training_seed,world_seed,group)]=row["return"]
                    if group=="trained":
                        differences[family]["vs_untrained"].setdefault(training_seed,{})[world_seed]=None
                        differences[family]["vs_heuristic"].setdefault(training_seed,{})[world_seed]=None
        # Use same world seed comparisons, do not treat replicated heuristic values as independent checkpoints.
        for training_seed in saved:
            for world_seed in world_seeds:
                differences[family]["vs_untrained"][training_seed][world_seed]=all_returns[(family,training_seed,world_seed,"trained")]-all_returns[(family,training_seed,world_seed,"untrained")]
                differences[family]["vs_heuristic"][training_seed][world_seed]=all_returns[(family,training_seed,world_seed,"trained")]-next(r["return"] for r in full if r["family"]==family and r["world_seed"]==world_seed and r["policy_group"]=="heuristic")
    summaries=[]
    for family,world_seeds in seeds.items():
        for baseline in ("untrained","heuristic"):
            diff=differences[family]["vs_"+baseline]
            stats=bootstrap_ci(diff,BOOTSTRAPS,seed=99231+sum(map(ord,family))+len(baseline))
            learned_returns=[all_returns[(family,s,w,"trained")] for s in saved for w in world_seeds]
            base_returns=([all_returns[(family,s,w,"untrained")] for s in saved for w in world_seeds] if baseline=="untrained" else [next(r["return"] for r in full if r["family"]==family and r["world_seed"]==w and r["policy_group"]=="heuristic") for w in world_seeds])
            baseline_mean=float(np.mean(base_returns))
            margins=pilot["decision_margins"][condition][family][baseline]
            utility_margin,harm_margin,margin_rule=comparison_margins(
                baseline_mean,margins["near_zero_utility_absolute_margin"])
            summaries.append({"family":family,"comparison":f"trained_vs_{baseline}","paired_return_difference":stats,
                "mean_trained_return":float(np.mean(learned_returns)),"mean_baseline_return":baseline_mean,
                "primary_margin_5pct_baseline_abs":utility_margin,
                "margin_rule":margin_rule,
                "primary_margin_met":bool(float(np.mean(learned_returns))-baseline_mean >= utility_margin),
                "positive_ci":stats["ci95_low"]>0,"decision_gate_pass":bool(stats["ci95_low"]>0 and (family!="primary_starter_14x10" or float(np.mean(learned_returns))-baseline_mean >= utility_margin)),
                "mean_trained_episode_return":float(np.mean(learned_returns)),"mean_baseline_episode_return":float(np.mean(base_returns))})
            summary=summaries[-1]
            summary["supported_harm_margin_10pct_baseline_abs"]=harm_margin
            summary["statistically_supported_harm_beyond_margin"]=stats["ci95_high"] < -harm_margin
            summary["no_supported_harm_gate_pass"]=not summary["statistically_supported_harm_beyond_margin"]
    # Group descriptive metrics for each cohort/family, all episodes retained.
    groups=[]
    for family in seeds:
        for group in ("trained","untrained","heuristic"):
            rows=[r for r in full if r["family"]==family and r["policy_group"]==group]
            action_counts=Counter();reason_counts=Counter()
            for r in rows:
                action_counts.update(r["action_counts"]);reason_counts.update(r["navigation_reason_counts"])
            groups.append({"family":family,"policy_group":group,"episodes":len(rows),"action_counts":dict(action_counts),
                "action_fractions":{a.value:action_counts[a.value]/(len(rows)*TICKS*5) for a in Action},
                "mean_episode_return":float(np.mean([r["return"] for r in rows])),"invalid_outputs":sum(r["invalid_outputs"] for r in rows),
                "navigation_reason_counts":dict(reason_counts),"mean_final_toy_physiology":{k:float(np.mean([r["final_toy_physiology"][k]["mean"] for r in rows])) for k in ("hunger","thirst","fatigue","social_need")}})
    test_output=output/condition/"locked-test-episodes.jsonl";test_output.parent.mkdir(parents=True,exist_ok=True)
    with test_output.open("w") as f:
        for row in full:f.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
    return {"comparisons":summaries,"cohort_summaries":groups,"episode_file":str(test_output.relative_to(ROOT)),"episode_file_sha256":sha256(test_output),"episodes":len(full)}


def run_precision_pilot(selected, initials, manifest, output: Path) -> dict[str, Any]:
    """Estimate variance/runtime only; no pilot policy ranking or selection."""
    pilot_seeds=manifest["seed_protocol"]["independent_precision_pilot_world_seeds"]
    training_seeds=manifest["seed_protocol"]["training_checkpoint_seed_ids"]
    all_metrics={}; started=__import__("time").perf_counter()
    for spec in manifest["experiment_conditions"]["confirmatory_learning_conditions"]:
        condition=spec["id"]; all_metrics[condition]={}
        for family in manifest["environment_families"]:
            family_id=family["id"]
            matrices={"vs_untrained":{},"vs_heuristic":{}}
            for sid in training_seeds: matrices["vs_untrained"][sid]={};matrices["vs_heuristic"][sid]={}
            heuristic_returns={}
            for world_seed in pilot_seeds:
                heuristic_returns[world_seed]=quick_return(HeuristicBaseline(),family_id,world_seed)
                for sid in training_seeds:
                    tr=quick_return(selected[condition][sid]["policy"],family_id,world_seed)
                    init=quick_return(initials[condition][sid],family_id,world_seed)
                    matrices["vs_untrained"][sid][world_seed]=tr-init
                    matrices["vs_heuristic"][sid][world_seed]=tr-heuristic_returns[world_seed]
            all_metrics[condition][family_id]={"matrices":matrices,"baseline_means":{
                "untrained":float(np.mean([quick_return(initials[condition][sid],family_id,s) for sid in training_seeds for s in pilot_seeds])),
                "heuristic":float(np.mean(list(heuristic_returns.values()))),
            },"pilot_paired_difference_std_by_comparison":{k:float(np.std([v for row in mat.values() for v in row.values()],ddof=1)) for k,mat in matrices.items()},
              "training_seed_cluster_sd_by_comparison":{k:float(np.std([np.mean(list(row.values())) for row in mat.values()],ddof=1)) for k,mat in matrices.items()}}
    # Counts are selected only from pilot precision estimates, before any locked
    # evaluation outcome is read. Use a common family count across both conditions
    # and both baseline contrasts; never reduce the preregistered minimum of 30.
    count_plan={}; precision_rows=[]
    for family in (f["id"] for f in manifest["environment_families"]):
        selected_n=30
        for condition in all_metrics:
            for baseline in ("untrained","heuristic"):
                pilot=all_metrics[condition][family]; mat=pilot["matrices"]["vs_"+baseline]
                base_mean=pilot["baseline_means"][baseline]
                # Predeclared near-zero fallback: 5% of pilot baseline-return SD.
                margin=(.05*abs(base_mean) if abs(base_mean)>=.1 else .05*max(.001,pilot["pilot_paired_difference_std_by_comparison"]["vs_"+baseline]))
                chosen=60
                for n_world in (30,40,50,60):
                    subset={sid:{s:mat[sid][s] for s in pilot_seeds[:n_world]} for sid in training_seeds}
                    ci=bootstrap_ci(subset,1000,seed=733+sum(map(ord,condition+family+baseline))+n_world)
                    half=(ci["ci95_high"]-ci["ci95_low"])/2
                    precision_rows.append({"condition":condition,"family":family,"baseline":baseline,"candidate_world_count":n_world,
                        "pilot_estimated_ci_half_width":half,"primary_utility_margin":margin,"target_half_width":.5*margin,
                        "meets_precision_target":half<=.5*margin})
                    if half<=.5*margin and chosen==60: chosen=n_world
                selected_n=max(selected_n,chosen)
        family_rows=[r for r in precision_rows if r["family"]==family]
        for row in family_rows: row["selected_locked_world_count"] = selected_n
        selected_rows=[r for r in family_rows if r["candidate_world_count"]==selected_n]
        count_plan[family]={"world_count":selected_n,"pilot_precise_at_selected_count":len(selected_rows)==len(all_metrics)*2 and all(
            r["meets_precision_target"] for r in selected_rows),
            "locked_seed_ids":(manifest["seed_protocol"]["locked_test_world_seeds_by_family"][family][:30] +
              (manifest["seed_protocol"]["predeclared_precision_reserve_world_seeds"][family][:selected_n-30] if selected_n>30 else []))}
    # Publish only variance/precision data (not pilot means, ranking, or action outcomes).
    runtime=__import__("time").perf_counter()-started
    decision_margins={}
    for condition, families in all_metrics.items():
        decision_margins[condition]={}
        for family, data in families.items():
            decision_margins[condition][family]={}
            for baseline in ("untrained","heuristic"):
                baseline_mean=data["baseline_means"][baseline]
                diff_sd=data["pilot_paired_difference_std_by_comparison"]["vs_"+baseline]
                near_zero_utility=.05*max(.001,diff_sd)
                decision_margins[condition][family][baseline]={
                    "pilot_baseline_mean":baseline_mean,
                    "pilot_paired_difference_sd":diff_sd,
                    "near_zero_utility_absolute_margin":near_zero_utility,
                    "near_zero_harm_absolute_margin":near_zero_utility,
                    "rule":"if absolute locked baseline mean < 0.1, use frozen pilot fallback; otherwise use 5% utility / 10% harm of locked baseline absolute mean"}
    document={"purpose":"runtime and interval precision only; no policy/condition/contrast selection",
      "pilot_world_seed_ids":pilot_seeds,"pilot_world_count_per_family":len(pilot_seeds),"training_checkpoint_count":len(training_seeds),
      "pilot_rollout_seconds":runtime,"paired_difference_sd_and_training_seed_cluster_sd":{
        condition:{family:{"paired_difference_sd":data["pilot_paired_difference_std_by_comparison"],
          "training_seed_cluster_sd":data["training_seed_cluster_sd_by_comparison"]} for family,data in fams.items()} for condition,fams in all_metrics.items()},
      "precision_candidates":precision_rows,"locked_sample_count_plan":count_plan,
      "decision_margins":decision_margins,
      "near_zero_baseline_margin_rule":"if pilot |baseline mean|<0.1, utility margin = 0.05 × max(0.001, pilot paired episode-difference SD)",
      "acceptance_target":"planned 95% crossed-bootstrap CI half-width <= 0.5 × 5% absolute baseline mean (or near-zero fallback); n in {30,40,50,60}; 10 training seeds retained",
      "all_precision_gates_met":all(v["pilot_precise_at_selected_count"] for v in count_plan.values())}
    dest=output/"pilot-precision-runtime.json";dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(document,indent=2,sort_keys=True)+"\n")
    document["output"]=str(dest.relative_to(ROOT));document["output_sha256"]=sha256(dest)
    return document


def freeze_analysis_protocol(output: Path, manifest: dict[str, Any], pilot: dict[str, Any],
                             training_results: dict[str, Any], checkpoint_hashes: dict[str, str],
                             training_summary_sha256: str) -> dict[str, Any]:
    """Persist exact source/seed/analysis contract before locked outcome access."""
    allowlist={}
    for condition,condition_result in training_results.items():
        allowlist[condition]={}
        for run in condition_result["training_seeds"]:
            seed=run["seed"]; prefix=str((output/condition/f"seed-{seed}").relative_to(ROOT))
            policy_path=prefix+".trained.npz"; sidecar_path=prefix+".trained.json"
            initial_path=prefix+".initial.npz"; initial_sidecar_path=prefix+".initial.json"
            allowlist[condition][str(seed)]={"checkpoint_path":policy_path,"checkpoint_sha256":checkpoint_hashes[policy_path],
              "sidecar_path":sidecar_path,"sidecar_sha256":checkpoint_hashes[sidecar_path],
              "initial_checkpoint_path":initial_path,"initial_checkpoint_sha256":checkpoint_hashes[initial_path],
              "initial_sidecar_path":initial_sidecar_path,"initial_sidecar_sha256":checkpoint_hashes[initial_sidecar_path],
              "selected_epoch":run["selected_episode"],"training_seed":seed}
    protocol={"status":"frozen_before_locked_test_outcomes",
      "manifest_sha256":sha256(MANIFEST),"training_eval_source_sha256":sha256(Path(__file__)),
      "training_summary_sha256":training_summary_sha256,
      "test_source_sha256":sha256(ROOT/"tests/test_train_eval_mvp1.py"),
      "candidate_training_seed_ids":manifest["seed_protocol"]["training_checkpoint_seed_ids"],
      "selected_checkpoint_allowlist":allowlist,
      "validation_seed_ids":manifest["seed_protocol"]["validation_world_seeds_for_checkpoint_selection_only"]["primary_starter_14x10"],
      "candidate_episodes":list(CHECKPOINT_EPISODES),"episodes_per_seed_condition":2000,
      "decision_margins":pilot["decision_margins"],
      "locked_sample_counts_by_family":{family:plan["world_count"] for family,plan in pilot["locked_sample_count_plan"].items()},
      "locked_seed_ids_by_family":{family:plan["locked_seed_ids"] for family,plan in pilot["locked_sample_count_plan"].items()},
      "ticks_per_episode":TICKS,"synthetic_days_per_episode":DAYS,"action_inference":"deterministic argmax",
      "bootstrap":"2000 two-way crossed checkpoint/world seed paired-cell resamples, percentile 95% CI; two-way cluster-robust sensitivity",
      "gate":manifest["outcomes_and_statistics"],"precision_pilot_sha256":pilot["output_sha256"],
      "evaluation_order":"training all seeds and validation selection; pilot-only precision/runtime; freeze this protocol; then locked outcomes",
      "prohibitions":manifest["prohibitions"]}
    dest=output/"frozen-analysis-protocol.json";dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(protocol,indent=2,sort_keys=True)+"\n")
    protocol["path"]=str(dest.relative_to(ROOT));protocol["sha256"]=sha256(dest)
    write_prelock_attestation(output,dest)
    return protocol


def write_prelock_attestation(output: Path, protocol_path: Path) -> dict[str, Any]:
    """Pin the exact frozen protocol bytes in a separate pre-outcome companion."""
    protocol=json.loads(protocol_path.read_text())
    companion={"status":"attested_before_locked_test_outcomes",
      "protocol_path":str(protocol_path.relative_to(ROOT)),"protocol_sha256":sha256(protocol_path),
      "training_eval_source_sha256":protocol["training_eval_source_sha256"],
      "test_source_sha256":protocol["test_source_sha256"],"manifest_sha256":protocol["manifest_sha256"],
      "training_summary_sha256":protocol["training_summary_sha256"],
      "precision_pilot_sha256":protocol["precision_pilot_sha256"]}
    dest=output/"pre-lock-attestation.json"
    dest.write_text(json.dumps(companion,indent=2,sort_keys=True)+"\n")
    companion["path"]=str(dest.relative_to(ROOT));companion["sha256"]=sha256(dest)
    return companion


def verify_prelock_attestation(protocol_path: Path, attestation_path: Path) -> dict[str, bool]:
    protocol=json.loads(protocol_path.read_text()); attestation=json.loads(attestation_path.read_text())
    protocol_digest=sha256(protocol_path)
    checks={"attestation_status":attestation.get("status")=="attested_before_locked_test_outcomes",
      "protocol_path":attestation.get("protocol_path")==str(protocol_path.relative_to(ROOT)),
      "protocol_digest":attestation.get("protocol_sha256")==protocol_digest}
    for key in ("training_eval_source_sha256","test_source_sha256","manifest_sha256","training_summary_sha256","precision_pilot_sha256"):
        checks[f"{key}_bound"]=attestation.get(key)==protocol.get(key)
    return checks


def verify_locked_cohort(output: Path, protocol: dict[str, Any], summary: dict[str, Any]) -> dict[str, bool]:
    """Fail closed on any selected or paired-initial cohort/provenance mismatch."""
    manifest=read_manifest(); expected_seeds=[int(s) for s in protocol["candidate_training_seed_ids"]]
    expected_conditions={item["id"] for item in manifest["experiment_conditions"]["confirmatory_learning_conditions"]}
    allowlist=protocol["selected_checkpoint_allowlist"]
    if set(allowlist)!=expected_conditions: raise ValueError("frozen selected-checkpoint condition set mismatch")
    checks={"condition_set":True,"complete_seed_matrix":True,"summary_checkpoint_map":True}
    summary_hashes=summary["checkpoint_sha256"]
    training=summary["training"]
    for condition in sorted(expected_conditions):
        seeds=allowlist[condition]
        if {int(s) for s in seeds}!=set(expected_seeds): raise ValueError(f"frozen seed matrix mismatch for {condition}")
        training_rows={int(row["seed"]):row for row in training[condition]["training_seeds"]}
        if set(training_rows)!=set(expected_seeds): raise ValueError(f"training summary seed matrix mismatch for {condition}")
        for seed in expected_seeds:
            item=seeds[str(seed)]; row=training_rows[seed]
            if int(item["training_seed"])!=seed or int(item["selected_epoch"])!=int(row["selected_episode"]):
                raise ValueError(f"selected checkpoint seed/epoch mismatch: {condition} seed {seed}")
            for path_key, digest_key in (("checkpoint_path","checkpoint_sha256"),("sidecar_path","sidecar_sha256"),
                                         ("initial_checkpoint_path","initial_checkpoint_sha256"),("initial_sidecar_path","initial_sidecar_sha256")):
                if path_key not in item or digest_key not in item: raise ValueError(f"frozen protocol missing {path_key}: {condition} seed {seed}")
                rel=item[path_key]; expected=item[digest_key]; file_path=ROOT/rel
                if summary_hashes.get(rel)!=expected: raise ValueError(f"training summary hash-map mismatch: {rel}")
                if not file_path.is_file() or sha256(file_path)!=expected: raise ValueError(f"artifact hash mismatch: {rel}")
            selected_meta=json.loads((ROOT/item["sidecar_path"]).read_text())["metadata"]
            initial_meta=json.loads((ROOT/item["initial_sidecar_path"]).read_text())["metadata"]
            for metadata,label in ((selected_meta,"selected"),(initial_meta,"initial")):
                if metadata.get("condition")!=condition or metadata.get("training_condition")!=condition or int(metadata.get("training_seed",-1))!=seed:
                    raise ValueError(f"{label} sidecar condition/seed mismatch: {condition} seed {seed}")
                if int(metadata.get("selected_episode",-1))!=int(item["selected_epoch"]):
                    raise ValueError(f"{label} sidecar selected epoch mismatch: {condition} seed {seed}")
            if selected_meta.get("candidate_validation_means")!={str(k):v for k,v in row["validation_mean_return_by_candidate"].items()}:
                raise ValueError(f"candidate validation score mismatch: {condition} seed {seed}")
    return checks


def persist_training_summary(phase, summary_path, training, checkpoint_hashes):
    """Write provenance only in phases that generated training artifacts.

    Evaluation phases consume this file as a frozen input. Re-serializing
    semantically identical JSON can still invalidate its byte-level hash.
    """
    if phase not in ("train", "all"):
        return False
    summary_path.write_text(json.dumps(
        {"training": training, "checkpoint_sha256": checkpoint_hashes},
        indent=2, sort_keys=True) + "\n")
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="artifacts/validation/mvp1-remediation/training-eval")
    parser.add_argument("--phase",choices=("train","evaluate","locked","all"),default="all")
    parser.add_argument("--family-limit",type=int,default=None,help="engineering smoke only; not valid for acceptance")
    parser.add_argument("--world-limit",type=int,default=None,help="engineering smoke only; not valid for acceptance")
    args=parser.parse_args()
    smoke = args.phase == "evaluate" and args.family_limit is not None and args.world_limit is not None
    if smoke and (args.family_limit <= 0 or args.world_limit <= 0 or args.family_limit > 1 or args.world_limit > 1):
        parser.error("local validation smoke is limited to one family and one world")
    try:
        require_kaggle_for_full_workload("MVP1 training/validation", smoke=smoke)
    except RuntimeError as exc:
        parser.error(str(exc))
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=read_manifest();seed_ids=manifest["seed_protocol"]["training_checkpoint_seed_ids"]
    trainworlds=manifest["seed_protocol"]["encoder_training_world_seeds"]
    train_results={}; selected={}; initials={}; hashes={}
    conditions=manifest["experiment_conditions"]["confirmatory_learning_conditions"]
    if args.phase in ("evaluate","locked"):
        summary_path=out/"training-summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError("run --phase train first; frozen training-summary.json is missing")
        saved_summary=json.loads(summary_path.read_text())
        train_results=saved_summary["training"]
        hashes=saved_summary["checkpoint_sha256"]
        for spec in conditions:
            condition=spec["id"]; cond_dir=out/condition
            selected[condition]={}; initials[condition]={}
            for seed in seed_ids:
                trained=load_checkpoint(cond_dir/f"seed-{seed}.trained.npz")
                initial=load_checkpoint(cond_dir/f"seed-{seed}.initial.npz")
                metadata=json.loads((cond_dir/f"seed-{seed}.trained.json").read_text())["metadata"]
                selected[condition][seed]={"policy":trained,"selected_episode":int(metadata["selected_episode"])}
                initials[condition][seed]=initial
    else:
        for spec in conditions:
            condition=spec["id"]; cond_dir=out/condition;cond_dir.mkdir(parents=True,exist_ok=True)
            samples=collect_encoder_states(trainworlds) if condition=="encoder_generated_wider_support_current_objective" else None
            if samples is not None:
                coverage={"feature_min":np.min(np.stack([x for x,_ in samples]),axis=0).tolist(),"feature_max":np.max(np.stack([x for x,_ in samples]),axis=0).tolist(),
                    "feature_nonzero_fraction":np.mean(np.stack([x for x,_ in samples])!=0,axis=0).tolist(),"sample_count":len(samples),"world_seeds":trainworlds}
                (cond_dir/"training-feature-coverage.json").write_text(json.dumps(coverage,indent=2)+"\n")
            selected[condition]={};initials[condition]={};seed_reports=[]
            for seed in seed_ids:
                if condition=="four_case_current_objective":candidates,initial,curve=train_four_case(seed,cond_dir,condition)
                else:candidates,initial,curve=train_encoder(seed,samples,cond_dir,condition)
                validation_seeds=manifest["seed_protocol"]["validation_world_seeds_for_checkpoint_selection_only"]["primary_starter_14x10"]
                scores={ep:float(np.mean([quick_return(policy,"primary_starter_14x10",s) for s in validation_seeds])) for ep,policy in candidates.items()}
                selected_episode=max(CHECKPOINT_EPISODES,key=lambda ep:(scores[ep],-ep)); trained=candidates[selected_episode]
                trained.episodes=selected_episode;initial.episodes=0
                prefix=f"seed-{seed}"
                meta={"condition":condition,"training_condition":condition,"frozen_manifest":str(MANIFEST.relative_to(ROOT)),"training_seed":seed,"selected_episode":selected_episode,
                      "candidate_validation_means":scores,"algorithm":"seeded NumPy tanh MLP REINFORCE; exact expected-reward state baseline",
                      "reward_schedule":{"correct":1.0,"incorrect":-0.25},"correct_action":"action attaining maximal unchanged one-step step_physiology toy reward; ties all correct"}
                for epoch,candidate in candidates.items():
                    candidate.episodes=epoch
                    candidate_path=cond_dir/f"{prefix}.epoch-{epoch}.npz"
                    candidate_meta={**meta,"candidate_episode":epoch,"candidate_validation_mean_return":scores[epoch],
                                    "selected_for_locked_evaluation":epoch==selected_episode}
                    save_checkpoint(candidate,candidate_path,metadata=candidate_meta)
                    hashes[str(candidate_path.relative_to(ROOT))]=sha256(candidate_path)
                    hashes[str(candidate_path.with_suffix('.json').relative_to(ROOT))]=sha256(candidate_path.with_suffix('.json'))
                tpath=cond_dir/f"{prefix}.trained.npz";ipath=cond_dir/f"{prefix}.initial.npz"
                save_checkpoint(trained,tpath,metadata=meta);save_checkpoint(initial,ipath,metadata={**meta,"role":"same-seed initialization comparator"})
                selected[condition][seed]={"policy":trained,"selected_episode":selected_episode};initials[condition][seed]=initial
                hashes[str(tpath.relative_to(ROOT))]=sha256(tpath);hashes[str(tpath.with_suffix('.json').relative_to(ROOT))]=sha256(tpath.with_suffix('.json'))
                hashes[str(ipath.relative_to(ROOT))]=sha256(ipath);hashes[str(ipath.with_suffix('.json').relative_to(ROOT))]=sha256(ipath.with_suffix('.json'))
                seed_reports.append({"seed":seed,"selected_episode":selected_episode,"validation_mean_return_by_candidate":scores,
                    "training_curve_first100_mean":float(np.mean(curve.get("batch_sampled_toy_reward",curve.get("episode_greedy_mean_reward",[]))[:100])),
                    "training_curve_last100_mean":float(np.mean(curve.get("batch_sampled_toy_reward",curve.get("episode_greedy_mean_reward",[]))[-100:])),
                    "parameter_delta_l2":float(np.linalg.norm(trained.parameter_vector()-initial.parameter_vector()))})
                (cond_dir/f"{prefix}-training-curves.json").write_text(json.dumps(curve,separators=(",",":"))+"\n")
            # Re-run one complete training seed before opening any locked test outcomes.
            # Compare raw parameters to its saved checkpoint; this is a reproducibility
            # check, not a policy-selection or test-seed query.
            if condition=="four_case_current_objective": replay,_,_=train_four_case(seed_ids[0],cond_dir,condition)
            else: replay,_,_=train_encoder(seed_ids[0],samples,cond_dir,condition)
            chosen=selected[condition][seed_ids[0]]["selected_episode"]
            saved_policy=load_checkpoint(cond_dir/f"seed-{seed_ids[0]}.trained.npz")
            training_reproducibility={"seed":seed_ids[0],"selected_episode":chosen,
                "bit_identical_parameters":bool(np.array_equal(replay[chosen].parameter_vector(),saved_policy.parameter_vector()))}
            train_results[condition]={"training_seeds":seed_reports,"episode_budget_per_seed":2000,"candidate_episodes":list(CHECKPOINT_EPISODES),
                "validation_seed_ids":manifest["seed_protocol"]["validation_world_seeds_for_checkpoint_selection_only"]["primary_starter_14x10"],"training_world_seed_ids":trainworlds if samples is not None else [],
                "training_samples":len(samples) if samples is not None else 4,"training_reproducibility_check":training_reproducibility}
    summary_path=out/"training-summary.json"
    persist_training_summary(args.phase, summary_path, train_results, hashes)
    if args.phase=="train":
        print(json.dumps({"training_summary":str(summary_path.relative_to(ROOT)),"sha256":sha256(summary_path),
                          "conditions":list(train_results)},indent=2));return
    # Pilot-derived precision calculation precedes the first locked outcome access.
    if args.phase=="locked":
        pilot_path=out/"pilot-precision-runtime.json"
        protocol_path=out/"frozen-analysis-protocol.json"
        if not pilot_path.is_file() or not protocol_path.is_file(): raise FileNotFoundError("locked phase requires prior pilot and frozen analysis protocol")
        pilot=json.loads(pilot_path.read_text());pilot["output"]=str(pilot_path.relative_to(ROOT));pilot["output_sha256"]=sha256(pilot_path)
        frozen_protocol=json.loads(protocol_path.read_text());frozen_protocol["path"]=str(protocol_path.relative_to(ROOT));frozen_protocol["sha256"]=sha256(protocol_path)
        checks={"source":frozen_protocol["training_eval_source_sha256"]==sha256(Path(__file__)),
                "tests":frozen_protocol["test_source_sha256"]==sha256(ROOT/"tests/test_train_eval_mvp1.py"),
                "manifest":frozen_protocol["manifest_sha256"]==sha256(MANIFEST),
                "training_summary":frozen_protocol["training_summary_sha256"]==sha256(summary_path),
                "pilot":frozen_protocol["precision_pilot_sha256"]==pilot["output_sha256"]}
        attestation_checks=verify_prelock_attestation(protocol_path,out/"pre-lock-attestation.json")
        checks.update({f"attestation_{key}":value for key,value in attestation_checks.items()})
        current_summary=json.loads(summary_path.read_text())
        try:
            cohort_checks=verify_locked_cohort(out,frozen_protocol,current_summary)
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError(f"frozen selected/initial cohort verification failed before locked rollout: {exc}") from exc
        checks.update({f"cohort_{key}":value for key,value in cohort_checks.items()})
        if not all(checks.values()): raise ValueError(f"frozen protocol hash verification failed: {checks}")
        frozen_protocol["hash_verification"]=checks
    else:
        pilot=run_precision_pilot(selected,initials,manifest,out)
        frozen_protocol=freeze_analysis_protocol(out,manifest,pilot,train_results,hashes,sha256(summary_path))
        if args.phase=="evaluate":
            print(json.dumps({"stage":"pilot_precision_complete_protocol_frozen","protocol":frozen_protocol,"pilot_precision_met":pilot["all_precision_gates_met"]},indent=2))
            return
    # Confirmatory locked split is only selected after training and pilot precision.
    results={}
    for spec in conditions:
        condition=spec["id"]
        m={**manifest}
        for family,plan in pilot["locked_sample_count_plan"].items():
            m["seed_protocol"]["locked_test_world_seeds_by_family"][family]=plan["locked_seed_ids"]
        if args.family_limit:m["seed_protocol"]["locked_test_world_seeds_by_family"]={k:v[:args.world_limit] if args.world_limit else v for k,v in list(m["seed_protocol"]["locked_test_world_seeds_by_family"].items())[:args.family_limit]}
        results[condition]=family_eval(condition,selected[condition],initials[condition],out,m,pilot)
    report={"title":"Preregistered synthetic learned-policy training and held-out evaluation","manifest_sha256":sha256(MANIFEST),
      "protocol_status":"FULL_PREREGISTERED" if not args.family_limit and not args.world_limit else "SMOKE_ONLY_NOT_ACCEPTANCE_EVIDENCE",
      "training":train_results,"pilot_precision_runtime":{k:v for k,v in pilot.items() if k not in ("output",)},"frozen_analysis_protocol":frozen_protocol,"evaluation":results,"checkpoint_sha256":hashes,"source_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),ROOT/"farm_sim/training.py",ROOT/"farm_sim/policy.py",ROOT/"farm_sim/simulation.py",ROOT/"farm_sim/navigation.py",ROOT/"farm_sim/perception.py",ROOT/"farm_sim/physiology.py")},
      "limitations":["synthetic software rewards and normalized toy drives are not validated cattle behavior or biology","untrained and trained are deterministic argmax policies; heuristic is deterministic baseline","precision threshold is preregistered but not estimated-adjusted; report interval and gate directly","per-tick outputs preserve all outcomes; no run is excluded"],
      "learning_gate": ("INCONCLUSIVE_UNDERPOWERED" if not pilot["all_precision_gates_met"] else "PASS" if all(
          all(c["decision_gate_pass"] for c in r["comparisons"] if c["family"]=="primary_starter_14x10") and
          any(all(c["decision_gate_pass"] for c in r["comparisons"] if c["family"]==fam) for fam in ("wide_grid_20x14","resource_scarce_relocated")) and
          all(c["no_supported_harm_gate_pass"] for c in r["comparisons"])
          for r in results.values()) else "FAIL")}
    (out/"training-evaluation-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"report":str((out/"training-evaluation-report.json").relative_to(ROOT)),"learning_gate":report["learning_gate"],"conditions":list(results),"episodes":{k:v["episodes"] for k,v in results.items()},"protocol_status":report["protocol_status"]},indent=2))

if __name__=="__main__": main()
