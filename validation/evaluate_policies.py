#!/usr/bin/env python3
"""Reproducible MVP1 policy evaluation; all rewards and drives are toy constructs."""
from __future__ import annotations
import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from farm_sim.model import Action, FEATURE_NAMES, MemoryEvent, Position, AnimalMemory, ToyPhysiology, ResourcePatch
from farm_sim.policy import HeuristicBaseline, load_checkpoint
import farm_sim.simulation as simulation_module
from farm_sim.simulation import Simulation, make_world
from farm_sim.perception import perceive, encode_features
from farm_sim.memory import retrieve

TICKS_PER_DAY = 24  # Explicit convention: phase feature has a 24-tick cycle.
DAYS = 3
STEPS = TICKS_PER_DAY * DAYS
PRIMARY_EVALUATION_SEEDS = [11, 29, 47, 83, 101]
HELD_OUT_SEEDS = [17, 23, 71]
POLICIES = {
    "untrained_seed11": lambda: load_checkpoint(ROOT / "artifacts/policies/cattle-policy-seed-11.initial.npz"),
    "trained_seed11": lambda: load_checkpoint(ROOT / "artifacts/policies/default-policy.npz"),
    "heuristic_baseline": HeuristicBaseline,
}


def valid_decision(action, probs):
    return action in Action and len(probs) == len(Action) and all(math.isfinite(p) and 0 <= p <= 1 for p in probs) and abs(sum(probs)-1) < 1e-6


def scenario_world(scenario, seed):
    if scenario == "starter_seed":
        return make_world(seed)
    if scenario == "wide_grid_stress":
        return make_world(seed, width=20, height=14)
    if scenario == "scarce_relocated_resources_stress":
        world = make_world(seed)
        # Leave one available forage patch and move the sole water point to a corner.
        # This is a constructed software stress case, not an ecological scenario.
        return replace(world,
            forage=(ResourcePatch(Position(1, 8), True), ResourcePatch(Position(2, 2), False), ResourcePatch(Position(3, 7), False)),
            water=(ResourcePatch(Position(12, 8), True),))
    raise ValueError(f"unknown scenario: {scenario}")


def evaluate(policy_name, seed, scenario="starter_seed"):
    sim = Simulation(scenario_world(scenario, seed), POLICIES[policy_name]())
    actions = Counter(); rewards=[]; invalid=0; requested=0; no_move=0
    nav_reasons = Counter()
    original_navigate = simulation_module.navigate
    def counted_navigate(*args, **kwargs):
        result = original_navigate(*args, **kwargs)
        nav_reasons[result.reason] += 1
        return result
    simulation_module.navigate = counted_navigate
    nav_action = {Action.WALK, Action.GRAZE, Action.DRINK, Action.HERD_INTERACT}
    phys_sum = Counter(); action_ticks = defaultdict(Counter)
    try:
        for _ in range(STEPS):
            before={a.animal_id:a.position for a in sim.world.animals}
            result=sim.step()
            tick_reward=0.0
            for a in result.world.animals:
                actions[a.action.value]+=1; action_ticks[a.animal_id][a.action.value]+=1
                tick_reward += a.last_reward
                for key in ("hunger","thirst","fatigue","social_need"):
                    phys_sum[key] += getattr(a.physiology,key)
                if a.action in nav_action:
                    requested += 1
                    if a.position == before[a.animal_id]: no_move += 1
                if not valid_decision(a.action,a.last_probabilities): invalid += 1
            rewards.append(tick_reward)
    finally:
        simulation_module.navigate = original_navigate
    animals=sim.world.animals
    state={k:{"mean_over_final_animals":sum(getattr(a.physiology,k) for a in animals)/len(animals),
              "min":min(getattr(a.physiology,k) for a in animals), "max":max(getattr(a.physiology,k) for a in animals)}
           for k in ("hunger","thirst","fatigue","social_need")}
    total=sum(rewards)
    return {"policy":policy_name,"scenario":scenario,"world_seed":seed,"ticks_per_day":TICKS_PER_DAY,"days":DAYS,"ticks":STEPS,
            "action_counts":dict(actions),"action_fractions":{a:round(n/(STEPS*len(animals)),6) for a,n in actions.items()},
            "per_animal_action_counts":{k:dict(v) for k,v in action_ticks.items()},
            "reward_trajectory_per_tick_all_animals": [round(x,6) for x in rewards],
            "cumulative_reward":round(total,6),"mean_summed_agent_reward_per_tick":round(total/STEPS,6),
            "toy_episode_success_positive_cumulative_reward":total>0,
            "invalid_actions_or_probability_vectors":invalid,
            "navigation_no_movement_proxy":{"navigation_routed_action_count":requested,"no_movement_count":no_move,
             "no_movement_fraction":round(no_move/requested,6) if requested else None,
             "definition":"navigation-routed action with unchanged position; may include at-target/no-target outcomes and is not a confirmed pathfinding failure"},
            "navigation_reason_counts":dict(nav_reasons),
            "confirmed_no_reachable_target_failures":nav_reasons["no_reachable_target"],
            "final_toy_physiology":state,
            "mean_toy_physiology_across_all_animal_ticks":{k:round(v/(STEPS*len(animals)),6) for k,v in phys_sum.items()},
            "final_positions":{a.animal_id:[a.position.x,a.position.y] for a in animals},
            "bounds_ok":all(0<=a.position.x<sim.world.width and 0<=a.position.y<sim.world.height for a in animals)}


def intervention_vectors():
    """Canonical controlled valid feature fixture plus one-feature counterfactuals."""
    # A fixed policy input contract fixture, with visible resources, moderate drives,
    # recent success memory; interventions alter one selected feature at a time.
    x=[0.35,0.7,0.25,0.45,1,1,0.2,0.2,1,1,0,0.25,1,1,0,0,0.8,0,1,1]
    cases=[]
    interventions={
      "memory_water_success_to_blocked":(12,1.0,0.0,14,0.0,1.0),
      "memory_forage_success_to_depleted":(13,1.0,0.0,15,0.0,1.0),
      "perception_water_visible_to_unavailable":(5,1.0,0.0,9,1.0,0.0),
      "perception_forage_visible_to_unavailable":(4,1.0,0.0,8,1.0,0.0),
      "thirst_low_to_high":(1,0.1,0.95,None,None,None),
      "hunger_low_to_high":(0,0.1,0.95,None,None,None),
    }
    for name, (i,lo,hi,j,jlo,jhi) in interventions.items():
        base=x.copy(); changed=x.copy(); base[i]=lo; changed[i]=hi
        if j is not None: base[j]=jlo; changed[j]=jhi
        cases.append((name,base,changed))
    return cases


def decisions(policy, feats):
    d=policy.decide(feats)
    return {"action":d.action.value,"probabilities":{a.value:round(p,8) for a,p in zip(Action,d.probabilities)}}


def encoded(world, animal_id="cow-1"):
    animal=next(a for a in world.animals if a.animal_id==animal_id)
    observation=perceive(world,animal)
    memories=retrieve(animal.memory,world.tick)
    return encode_features(observation,memories,animal,world.width,world.height)


def production_intervention_worlds():
    """Counterfactual states passed through the actual perception/memory encoder."""
    world=make_world(991)
    animals=list(world.animals)
    # Controlled fixture places one inspected animal near both resources. This is a
    # constructed encoder test state, not a rollout or a naturalistic scenario.
    animals[0]=replace(animals[0],position=Position(2,2),physiology=ToyPhysiology(.35,.7,.25,.45),
        memory=AnimalMemory((MemoryEvent(0,"water_access",True,Position(4,2)),
                             MemoryEvent(0,"forage_patch",True,Position(3,2)))))
    world=replace(world,animals=tuple(animals),tick=1,
        forage=(ResourcePatch(Position(3,2),True),),water=(ResourcePatch(Position(4,2),True),))
    cases=[]
    base_animals=list(world.animals)
    # Memory: only event outcome changes; event kind, age and location are fixed.
    def with_memory(kind, success):
        animals=list(base_animals); a=animals[0]
        previous=[e for e in a.memory.events if e.kind != kind]
        event=MemoryEvent(0,kind,success,Position(4,2) if kind=="water_access" else Position(3,2))
        animals[0]=replace(a,memory=AnimalMemory(tuple(previous+[event])))
        return replace(world,animals=tuple(animals))
    cases.append(("encoder_memory_water_success_vs_blocked",world,with_memory("water_access",False)))
    cases.append(("encoder_memory_forage_success_vs_depleted",world,with_memory("forage_patch",False)))
    # Availability toggles preserve resource position/visibility.
    cases.append(("encoder_visible_water_available_vs_unavailable",world,replace(world,water=(ResourcePatch(Position(4,2),False),))))
    cases.append(("encoder_visible_forage_available_vs_depleted",world,replace(world,forage=(ResourcePatch(Position(3,2),False),))))
    # Relocation changes visible flag and distance jointly; call out this confound.
    cases.append(("encoder_water_relocated_out_of_view",world,replace(world,water=(ResourcePatch(Position(12,8),True),))))
    # Physiology intervention holds the entire world/resources/memory fixed.
    animals=list(base_animals); animals[0]=replace(animals[0],physiology=replace(animals[0].physiology,thirst=.1))
    low=replace(world,animals=tuple(animals))
    animals=list(base_animals); animals[0]=replace(animals[0],physiology=replace(animals[0].physiology,thirst=.95))
    high=replace(world,animals=tuple(animals))
    cases.append(("encoder_thirst_low_vs_high",low,high))
    return cases


def causal():
    out=[]
    for policy_name in POLICIES:
        policy=POLICIES[policy_name]()
        for name,base,changed in intervention_vectors():
            before=decisions(policy,base); after=decisions(policy,changed)
            out.append({"policy":policy_name,"intervention":name,"before":before,"after":after,
                        "argmax_changed":before["action"]!=after["action"],
                        "probability_delta":{a.value:round(after["probabilities"][a.value]-before["probabilities"][a.value],8) for a in Action}})
    return {"fixed_vector_interventions":out,
      "production_input_pathway_interventions":[
        {"policy":policy_name,"intervention":name,
         "before_features":dict(zip(FEATURE_NAMES, [round(v,6) for v in encoded(before_world)])),
         "after_features":dict(zip(FEATURE_NAMES, [round(v,6) for v in encoded(after_world)])),
         "changed_feature_names":[FEATURE_NAMES[i] for i,(a,b) in enumerate(zip(encoded(before_world),encoded(after_world))) if abs(a-b)>1e-9],
         "before":decisions(POLICIES[policy_name](),encoded(before_world)),
         "after":decisions(POLICIES[policy_name](),encoded(after_world)),
         "argmax_changed":decisions(POLICIES[policy_name](),encoded(before_world))["action"] != decisions(POLICIES[policy_name](),encoded(after_world))["action"],
         "evidence_scope":"policy-input pathway only: actual perceive/retrieve/encode_features then decide; not a whole-simulation counterfactual rollout",
         "joint_factor_note":"water relocation changes visibility and distance jointly; other paired cases target the named factor(s) while holding constructed fixture factors fixed"}
        for policy_name in POLICIES for name,before_world,after_world in production_intervention_worlds()]}


def main():
    rows=[evaluate(p,s) for s in PRIMARY_EVALUATION_SEEDS+HELD_OUT_SEEDS for p in POLICIES]
    # Structural stress scenarios are explicitly synthetic and are paired across policies.
    stress_scenarios={"wide_grid_stress":211,"scarce_relocated_resources_stress":307}
    rows.extend(evaluate(p,seed,scenario) for scenario,seed in stress_scenarios.items() for p in POLICIES)
    # Aggregate comparable paired-world-seed outcomes by policy and seed family.
    aggregates=[]
    for family,seeds in (("primary_evaluation_seeds",PRIMARY_EVALUATION_SEEDS),("held_out_evaluation_seeds",HELD_OUT_SEEDS)):
        for p in POLICIES:
            group=[r for r in rows if r["policy"]==p and r["world_seed"] in seeds and r["scenario"]=="starter_seed"]
            aggregates.append({"seed_family":family,"policy":p,"runs":len(group),
             "mean_cumulative_reward":round(sum(r["cumulative_reward"] for r in group)/len(group),6),
             "positive_reward_runs":sum(r["toy_episode_success_positive_cumulative_reward"] for r in group),
             "mean_invalid_actions":sum(r["invalid_actions_or_probability_vectors"] for r in group)/len(group),
             "mean_navigation_no_movement_fraction":round(sum(r["navigation_no_movement_proxy"]["no_movement_fraction"] for r in group)/len(group),6),
             "mean_final_toy_drives":{k:round(sum(r["final_toy_physiology"][k]["mean_over_final_animals"] for r in group)/len(group),6) for k in ("hunger","thirst","fatigue","social_need")}})
    for scenario in stress_scenarios:
        for p in POLICIES:
            group=[r for r in rows if r["policy"]==p and r["scenario"]==scenario]
            aggregates.append({"seed_family":"synthetic_structural_stress","scenario":scenario,"policy":p,"runs":len(group),
             "mean_cumulative_reward":round(sum(r["cumulative_reward"] for r in group)/len(group),6),
             "positive_reward_runs":sum(r["toy_episode_success_positive_cumulative_reward"] for r in group),
             "mean_invalid_actions":sum(r["invalid_actions_or_probability_vectors"] for r in group)/len(group),
             "mean_confirmed_no_reachable_target_failures":sum(r["confirmed_no_reachable_target_failures"] for r in group)/len(group),
             "mean_navigation_no_movement_fraction":round(sum(r["navigation_no_movement_proxy"]["no_movement_fraction"] for r in group)/len(group),6),
             "mean_final_toy_drives":{k:round(sum(r["final_toy_physiology"][k]["mean_over_final_animals"] for r in group)/len(group),6) for k in ("hunger","thirst","fatigue","social_need")}})
    doc={"title":"Pasture Studies MVP 1 policy evaluation","generated_by":"validation/evaluate_policies.py",
      "reproducible_command":"python3 validation/evaluate_policies.py",
      "scope":{"ticks_per_day":TICKS_PER_DAY,"ticks_convention":"24 discrete simulation ticks per synthetic day; this is an evaluation unit, not a biological timescale","days_per_run":DAYS,"ticks_per_run":STEPS,"primary_evaluation_world_seeds":PRIMARY_EVALUATION_SEEDS,"held_out_evaluation_world_seeds":HELD_OUT_SEEDS,"structural_stress_scenarios":stress_scenarios,"policies":list(POLICIES),"identical_world_seeds_paired":True,"all_policy_inference_deterministic":True,"checkpoint_training_task":"four-case one-step paired-memory task; evaluation world seeds were not training worlds"},
      "scientific_scope":"Toy software-task rewards and normalized drives are not empirically validated cattle physiology, behavior, welfare, or cognition. Held-out world seeds are not an empirical generalization benchmark.",
      "aggregate_results":aggregates,"runs":rows,"controlled_feature_interventions":causal(),
      "limitations":["Initial/untrained comparator is seed-11 initialization; trained policy is packaged default trained checkpoint.","Simulation policy decide is argmax deterministic inference; no stochastic policy sampling is used.","Navigation reason counts are captured by a temporary validation wrapper around the simulator's existing navigate function; no product behavior is changed.","The no-movement proxy remains distinct from exact no_reachable_target navigation failures and also includes at-target/no-target actions.","Feature counterfactuals, including production encoder-path tests, probe the policy-input pathway only and do not establish whole-simulation causal effects.","One encoder-path water relocation intervention jointly changes visibility and distance; it is labeled accordingly.","Positive cumulative toy reward is a reporting convention, not a validated success threshold.","Physiology summaries describe only normalized toy variables and synthetic transition equations.","Different starter seeds and the two structural stress worlds are synthetic software stress tests, not ecological generalization or husbandry validation."]}
    target=ROOT/".omx/research/mvp1-policy-evaluation.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(target.relative_to(ROOT)),"runs":len(rows),"ticks_per_run":STEPS,"total_policy_ticks":len(rows)*STEPS,
       "aggregates":aggregates,"fixed_vector_interventions":len(doc["controlled_feature_interventions"]["fixed_vector_interventions"]),
       "production_input_interventions":len(doc["controlled_feature_interventions"]["production_input_pathway_interventions"]),
       "invalid_action_total":sum(r["invalid_actions_or_probability_vectors"] for r in rows),
       "no_reachable_target_total":sum(r["confirmed_no_reachable_target_failures"] for r in rows)},indent=2))

if __name__=="__main__": main()
