"""Train and compare optional Stable-Baselines3 PPO pilots (synthetic task only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from farm_sim.gym_env import PastureEnv


def _evaluate(model, *, joint: bool, seed: int, episodes: int) -> dict:
    env = PastureEnv(joint=joint)
    returns = []
    per_animal = {f"cow-{index}": [] for index in range(1, 6)}
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        env.action_space.seed(seed + episode)
        done = False
        total = 0.0
        animal_totals = {animal: 0.0 for animal in per_animal}
        while not done:
            action = env.action_space.sample() if model is None else model.predict(observation, deterministic=True)[0]
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            for animal, animal_reward in info["individual_rewards"].items():
                animal_totals[animal] += animal_reward
            done = terminated or truncated
        returns.append(total)
        for animal, value in animal_totals.items():
            per_animal[animal].append(value)
    env.close()
    return {"mean_return": sum(returns) / len(returns), "episode_returns": returns,
            "mean_return_by_animal": {animal: sum(values) / len(values) for animal, values in per_animal.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sb3-ppo"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    args = parser.parse_args()
    if args.timesteps <= 0 or args.evaluation_episodes <= 0:
        parser.error("timesteps and evaluation-episodes must be positive")
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install the optional training stack with: python -m pip install -r requirements-rl.txt") from exc

    args.output.mkdir(parents=True, exist_ok=True)
    report = {"algorithm": "Stable-Baselines3 PPO", "seed": args.seed,
              "timesteps_per_mode": args.timesteps, "evaluation_episodes": args.evaluation_episodes,
              "environment": "PastureEnv; synthetic physiology/rewards; no biological claim", "modes": {}}
    for mode, joint in (("one-animal", False), ("five-animal-joint", True)):
        env = PastureEnv(joint=joint)
        env.reset(seed=args.seed)
        model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0)
        model.learn(total_timesteps=args.timesteps)
        checkpoint = args.output / f"ppo-{mode}.zip"
        model.save(checkpoint)
        report["modes"][mode] = {
            "observation_shape": list(env.observation_space.shape),
            "action_space": str(env.action_space),
            "random_policy": _evaluate(None, joint=joint, seed=args.seed + 10_000, episodes=args.evaluation_episodes),
            "ppo": _evaluate(model, joint=joint, seed=args.seed + 10_000, episodes=args.evaluation_episodes),
            "checkpoint": str(checkpoint),
        }
        env.close()
    report_path = args.output / "comparison-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
