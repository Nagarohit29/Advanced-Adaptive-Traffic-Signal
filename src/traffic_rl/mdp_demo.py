import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from traffic_rl.agents.q_learning import Discretizer, QLearningAgent
from traffic_rl.envs.base_env import APPROACHES, GREEN_APPROACHES, Observation
from traffic_rl.envs.simple_env import SimpleEnvConfig, SimpleTrafficEnv


@dataclass(frozen=True)
class StateBreakdown:
    state_index: int
    queues: tuple[int, ...]
    queue_bins: tuple[int, ...]
    phase: int
    phase_name: str
    elapsed: int
    elapsed_bin: int


def decompose_state(obs: Observation, discretizer: Discretizer) -> StateBreakdown:
    queues = tuple(int(q) for q in obs[: len(APPROACHES)])
    queue_bins = tuple(int(b) for b in np.digitize(obs[: len(APPROACHES)], discretizer.queue_edges))
    phase = int(obs[len(APPROACHES)])
    phase_name = "NS Green" if phase == 0 else "EW Green"
    elapsed = int(obs[len(APPROACHES) + 1])
    elapsed_bin = int(np.digitize(elapsed, discretizer.elapsed_edges))
    state_index = discretizer(obs)
    return StateBreakdown(
        state_index=state_index,
        queues=queues,
        queue_bins=queue_bins,
        phase=phase,
        phase_name=phase_name,
        elapsed=elapsed,
        elapsed_bin=elapsed_bin,
    )


def evaluate_bellman_step(
    env: SimpleTrafficEnv,
    agent: QLearningAgent,
    obs: Observation,
    n_samples: int = 200,
) -> dict[str, Any]:
    state_idx = agent.discretizer(obs)
    learned_q = [float(agent.q[state_idx, a]) for a in range(len(GREEN_APPROACHES))]
    action_evaluations = []

    queues_orig = env.queues.copy()
    phase_orig = env.phase
    elapsed_orig = env.elapsed
    sim_time_orig = env.sim_time

    for a in range(len(GREEN_APPROACHES)):
        rewards: list[float] = []
        next_max_qs: list[float] = []
        for _ in range(n_samples):
            env.queues = queues_orig.copy()
            env.phase = phase_orig
            env.elapsed = elapsed_orig
            env.sim_time = sim_time_orig
            next_obs, r, _, _, _ = env.step(a)
            next_idx = agent.discretizer(next_obs)
            rewards.append(r)
            next_max_qs.append(float(agent.q[next_idx].max()))

        mean_r = float(np.mean(rewards))
        mean_next_v = float(np.mean(next_max_qs))
        expected_bellman = mean_r + agent.gamma * mean_next_v
        action_evaluations.append(
            {
                "action": a,
                "action_name": "HOLD" if a == phase_orig else "SWITCH",
                "immediate_reward": mean_r,
                "continuation_value": mean_next_v,
                "bellman_value": expected_bellman,
                "learned_q": learned_q[a],
            }
        )

    env.queues = queues_orig
    env.phase = phase_orig
    env.elapsed = elapsed_orig
    env.sim_time = sim_time_orig

    return {
        "state_index": state_idx,
        "learned_q": learned_q,
        "best_action": int(np.argmax(learned_q)),
        "actions": action_evaluations,
    }


def run_mdp_demonstration(steps: int = 6, seed: int = 42) -> None:
    env = SimpleTrafficEnv(SimpleEnvConfig())
    model_path = Path("results/models/q_learning.npz")
    if model_path.exists():
        agent = QLearningAgent.load(model_path)
    else:
        discretizer = Discretizer([3, 7, 13], [10, 30])
        agent = QLearningAgent(discretizer, alpha=0.1, gamma=0.95, seed=seed)

    obs, _ = env.reset(seed=seed)
    print("=" * 70)
    print("FINITE MARKOV DECISION PROCESS (MDP) DEMO: ADAPTIVE TRAFFIC SIGNAL")
    print("=" * 70)
    print(f"Total Discrete States: {agent.discretizer.n_states}")
    print(f"Discount Factor gamma: {agent.gamma}")
    print("Queue Bins: <3, 3-6, 7-12, >=13 | Elapsed Bins: <10, 10-29, >=30")
    print("-" * 70)

    for step in range(steps):
        state_info = decompose_state(obs, agent.discretizer)
        bellman = evaluate_bellman_step(env, agent, obs, n_samples=50)
        action = agent.act(obs, greedy=True)
        act_name = "HOLD" if action == state_info.phase else "SWITCH"

        print(
            f"\n[Step {step + 1}] Sim Time: {env.sim_time}s | State Index: {state_info.state_index}"
        )
        print(
            f"  Queues (N,S,E,W): {state_info.queues} (Bins: {state_info.queue_bins}) | "
            f"Phase: {state_info.phase_name} | Green Elapsed: {state_info.elapsed}s"
        )
        print("  Bellman Action Values: ")
        for act_data in bellman["actions"]:
            print(
                f"    Action {act_data['action']} ({act_data['action_name']}): "
                f"E[R]={act_data['immediate_reward']:.3f}, "
                f"E[V(s')]={act_data['continuation_value']:.3f}, "
                f"Bellman RHS={act_data['bellman_value']:.3f}, "
                f"Learned Q={act_data['learned_q']:.3f}"
            )
        print(f"  Selected Action: {action} ({act_name})")

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state_info = decompose_state(next_obs, agent.discretizer)

        print(
            f"  Transition Result -> Reward: {reward:.3f} "
            f"(queue-sec: {info['queue_seconds']}, switched: {info['switched']})"
        )
        print(
            f"  Next State Index: {next_state_info.state_index} | "
            f"Queues: {next_state_info.queues} | Phase: {next_state_info.phase_name}"
        )

        obs = next_obs
        if terminated or truncated:
            break

    print("\n" + "=" * 70)
    print("MDP DEMONSTRATION COMPLETE")
    print("=" * 70)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Finite Markov Decision Process (MDP) Interactive Demonstration"
    )
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    run_mdp_demonstration(steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
