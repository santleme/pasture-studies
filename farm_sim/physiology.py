"""Pure, deliberately non-biological toy state/reward transition."""
from __future__ import annotations

from dataclasses import replace

from .model import Action, AnimalState, ToyPhysiology, clamp01


def step_physiology(state: ToyPhysiology, action: Action, *, forage_available: bool | None = None,
                    water_available: bool | None = None, herd_nearby: bool = False) -> tuple[ToyPhysiology, float, tuple[str, ...]]:
    """Return updated toy drives, invented scalar reward, and event labels.

    The coefficients encode a small software task only; they are not cattle science.
    """
    hunger, thirst, fatigue, social = state.hunger, state.thirst, state.fatigue, state.social_need
    hunger = clamp01(hunger + 0.025)
    thirst = clamp01(thirst + 0.03)
    fatigue = clamp01(fatigue + (0.025 if action in (Action.WALK, Action.GRAZE, Action.DRINK) else -0.015))
    social = clamp01(social + (0.015 if not herd_nearby else -0.01))
    reward = -0.015
    events: list[str] = []
    if action == Action.GRAZE:
        if forage_available is True:
            hunger = clamp01(hunger - 0.22); reward += 0.32; events.append("forage_success")
        elif forage_available is False:
            reward -= 0.12; events.append("forage_depleted")
        else:
            # En route to a patch: action intent is recorded, but not mistaken for failure.
            events.append("grazing_en_route")
    elif action == Action.DRINK:
        if water_available is True:
            thirst = clamp01(thirst - 0.3); reward += 0.42; events.append("water_success")
        elif water_available is False:
            reward -= 0.15; events.append("water_blocked")
        else:
            events.append("drinking_en_route")
    elif action == Action.REST:
        fatigue = clamp01(fatigue - 0.18); reward += 0.08; events.append("rested")
    elif action == Action.RUMINATE:
        hunger = clamp01(hunger - 0.025); fatigue = clamp01(fatigue - 0.06); reward += 0.025; events.append("ruminated")
    elif action == Action.HERD_INTERACT:
        if herd_nearby:
            social = clamp01(social - 0.2); reward += 0.1; events.append("herd_contact")
        else:
            reward -= 0.04; events.append("herd_alone")
    elif action == Action.WALK:
        reward -= 0.01; events.append("walked")
    # Small synthetic drive penalty makes reward responsive to state, without implying welfare.
    reward -= 0.02 * (hunger + thirst + fatigue + social)
    return ToyPhysiology(hunger, thirst, fatigue, social), float(reward), tuple(events)
