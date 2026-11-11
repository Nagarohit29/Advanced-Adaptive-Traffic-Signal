# Architecture

## Components

```
configs/*.yaml ──► train.py / evaluate.py ◄── app/streamlit_app.py
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
     Environment      Agent         Baseline
   (Gymnasium Env)  (Q-learning,   (fixed timer)
    simple_env /     SARSA, DQN)
    sumo_env
          │
          ▼
   episode loop ──► metrics CSV ──► plots / GIF / evaluation table
```

| Component | Responsibility | Depends on |
|---|---|---|
| `envs/base_env.py` | Shared spaces, observation builder and info keys | gymnasium, numpy |
| `envs/simple_env.py` | Custom simulator: Poisson arrivals, saturation-flow departures, yellow lost time | base_env |
| `envs/sumo_env.py` | Optional SUMO backend through TraCI; subclasses `SimpleTrafficEnv` | `sumo` extra |
| `episode.py` | The one episode loop and metric tally, shared by training, evaluation, the GIF and the app | envs |
| `agents/q_learning.py`, `agents/sarsa.py` | Discretizer, Q-table, TD update; SARSA overrides only the bootstrap target | numpy |
| `agents/dqn.py` | Stable-Baselines3 DQN on scaled observations | `dqn` extra, imported lazily |
| `baselines/fixed_timer.py` | Fixed timer, and reading the tuned timer back from `evaluation.csv` | none |
| `train.py` | Trains an agent from `configs/<agent>.yaml`; writes metrics, checkpoints and the model | envs, agents |
| `evaluate.py` | Tunes the timer on training seeds, scores every model on held-out seeds, writes `evaluation.csv` and plots | envs, agents, baselines |
| `demo.py` | Renders `results/demo.gif`: the agent beside the tuned timer | episode, plotting |
| `utils/` | CSV logging, plots, GIF frames | matplotlib |
| `app/streamlit_app.py` | Deployed demo: the committed Q-table and both timers on visitor-chosen demand | core package, streamlit |
| `scripts/run_full_pipeline.sh` | One command: train, evaluate, render the GIF | the CLIs above |

Agents and baselines never import an environment module. They see only observations and
actions, so the simulator can be swapped without touching them. Every controller goes
through the same `run_episode`, so all reported metrics come from one piece of code.

## Data flow

1. `train.py` reads a config and runs `run_episode` with exploration on. Each transition
   `(s, a, r, s', a')` goes to the agent's `update`. Per-episode metrics go to
   `results/runs/<agent>/metrics.csv` (git-ignored), and the final model to
   `results/models/` (committed, 25–100 KB each).
2. `evaluate.py` picks the fixed-timer split by grid search on training seeds 0–9, then
   runs every controller greedily on held-out seeds 1000–1009. It writes
   `results/evaluation.csv` (including the tuned green times) and both plots.
3. `demo.py` and the app read the tuned timer back from `evaluation.csv` instead of
   re-tuning, so the GIF, the app and the results table all use the same baseline.

## Deployment decision

**Platform: Streamlit Community Cloud.** Fallback: GitHub Pages.

Free options compared on 2026-10-27, from each provider's own docs:

| Option | Free? | Fit |
|---|---|---|
| Hugging Face Spaces (Gradio/Docker) | No. Creating them now needs a PRO plan; only static Spaces are free | Not usable |
| Hugging Face Spaces (static) | Yes | Results page only, no live simulation |
| Streamlit Community Cloud | Yes: about 2 vCPU and 2.7 GB RAM, sleeps after 12 h without traffic | Interactive Python demo, deploys from this public GitHub repo, redeploys on push |
| GitHub Pages | Yes | Static results page and GIF; no server-side Python |

Streamlit Cloud is the only free option that can run the trained agent live. The app
installs the pinned streamlit, gymnasium and numpy from `app/requirements.txt` and loads
the package straight from `src/` (no torch, no SUMO), which keeps it well
under the memory limit. If Streamlit Cloud can't be used, the fallback is a static GitHub
Pages page showing the plots, the GIF and the results table.

## CI/CD

- **Checks:** `scripts/check.sh` runs ruff (lint and format), mypy, pytest and a 5-episode
  smoke training run. It is the single definition of "passing".
- **Where it runs:** a local `.git/hooks/pre-push` runs `check.sh` and blocks any push
  that fails. GitHub Actions (`.github/workflows/ci.yml`) runs the same script, with
  read-only permissions and no secrets, but Actions has not been able to run on this
  account. So the pre-push hook is the gate that is actually enforced.
- **CD:** Streamlit Community Cloud redeploys the app from `main` on every push. Live at
  https://adaptive-traffic-signal-control-8x5zgibkmr432uaitwwtf7.streamlit.app. Free apps
  sleep after 12 hours without traffic, and the first visit after that takes a moment to
  wake.
- **Dependabot:** monthly dependency update PRs.

## Implemented and future work

Implemented and measured:

- Custom simulator behind the Gymnasium API, with contract tests
- Optional SUMO backend through TraCI, with matching spaces and reward
- Fixed-timer baseline, default and tuned (on training seeds only)
- Tabular Q-learning, SARSA, and DQN (Stable-Baselines3), all trained from YAML configs
- Held-out evaluation, learning-curve and comparison plots, demo GIF
- One-command pipeline, which reproduces every committed result byte for byte from a
  fresh clone
- Streamlit demo app with bounded inputs

Not done, and why:

- Multi-intersection coordination: out of scope for the single-intersection MDP
  (REQUIREMENTS.md)
- PPO / actor-critic: judged overkill for a 6-value state and 2 actions (PROJECT_PLAN §3)
- Evaluating the trained agents inside SUMO: SUMO counts queues differently, so the
  agents would need retraining there to be compared fairly
- Multi-seed DQN, and a γ-matched comparison with the tabular agents: DQN was trained
  once with γ = 0.99
- Turning movements, pedestrians, real-world demand data
