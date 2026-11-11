import numpy as np

from traffic_rl.agents.q_learning import Discretizer, QLearningAgent
from traffic_rl.envs.simple_env import SimpleTrafficEnv
from traffic_rl.mdp_demo import (
    decompose_state,
    evaluate_bellman_step,
    main,
    run_mdp_demonstration,
)


def test_decompose_state() -> None:
    discretizer = Discretizer([3, 7, 13], [10, 30])
    obs = np.array([2, 5, 8, 15, 1, 25], dtype=np.float32)
    breakdown = decompose_state(obs, discretizer)
    assert breakdown.queues == (2, 5, 8, 15)
    assert breakdown.queue_bins == (0, 1, 2, 3)
    assert breakdown.phase == 1
    assert breakdown.phase_name == "EW Green"
    assert breakdown.elapsed == 25
    assert breakdown.elapsed_bin == 1
    assert 0 <= breakdown.state_index < discretizer.n_states


def test_evaluate_bellman_step() -> None:
    env = SimpleTrafficEnv()
    discretizer = Discretizer([3, 7, 13], [10, 30])
    agent = QLearningAgent(discretizer, alpha=0.1, gamma=0.95, seed=0)
    obs = np.array([1, 1, 4, 4, 0, 10], dtype=np.float32)

    result = evaluate_bellman_step(env, agent, obs, n_samples=10)
    assert "state_index" in result
    assert "learned_q" in result
    assert len(result["actions"]) == 2
    for action_data in result["actions"]:
        assert "immediate_reward" in action_data
        assert "continuation_value" in action_data
        assert "bellman_value" in action_data


def test_run_mdp_demonstration(capsys) -> None:  # type: ignore[no-untyped-def]
    run_mdp_demonstration(steps=2, seed=1)
    captured = capsys.readouterr()
    assert "FINITE MARKOV DECISION PROCESS" in captured.out
    assert "Step 1" in captured.out
    assert "MDP DEMONSTRATION COMPLETE" in captured.out


def test_mdp_demo_main(capsys) -> None:  # type: ignore[no-untyped-def]
    main(["--steps", "1", "--seed", "123"])
    captured = capsys.readouterr()
    assert "Step 1" in captured.out
