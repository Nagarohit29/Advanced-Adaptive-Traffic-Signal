# MDP Formulation and Algorithm Choice

This document defines the decision problem the agent solves and explains why the project
uses tabular Q-learning as its primary method. Every value below matches the code in
`src/traffic_rl/` and `configs/q_learning.yaml`. If they ever disagree, the code is right
and this file is out of date.

## 1. Problem and simulator

One signal controls a four-way intersection. The goal is to keep vehicles from waiting,
while traffic demand shifts between the two directions over the course of an hour.

The primary environment is a custom simulator (`envs/simple_env.py`). It has no SUMO
dependency, so CI, the deployed demo and the core install can all run it.

- Four approaches, N, S, E, W. Each has one lane and carries through traffic only.
- Time advances in 1-second ticks.
- Arrivals: each tick, approach `i` receives `Poisson(λ_i)` vehicles, where `λ_i` comes
  from the demand schedule below.
- Departures: each approach with a green light and a non-empty queue releases one vehicle
  per tick with probability `saturation_flow = 0.5`. That is 1800 veh/h, the standard
  single-lane saturation flow from the Highway Capacity Manual.
- Two phases: `0` = NS green / EW red, `1` = EW green / NS red.
- Yellow lost time: changing phase inserts `yellow_time = 3` s with no departures on any
  approach. Arrivals continue during it.

Default demand schedule (vehicles per second, per approach):

| Window (s) | N | S | E | W | Dominant direction |
|---|---|---|---|---|---|
| 0 – 1200 | 0.16 | 0.16 | 0.06 | 0.06 | NS |
| 1200 – 2400 | 0.11 | 0.11 | 0.11 | 0.11 | balanced |
| 2400 – 3600 | 0.06 | 0.06 | 0.16 | 0.16 | EW |

Total arrivals average 0.44 veh/s in every window. Each window needs 0.44 of the cycle as
green, before yellow lost time (for example `0.16/0.5 + 0.06/0.5`). Demand is therefore
undersaturated, and queues stay bounded under any reasonable controller. Because the
dominant direction flips, no single fixed green split suits all three windows. That is
the case adaptive control is meant for.

A SUMO backend (`envs/sumo_env.py`, installed with the `sumo` extra) is an optional
cross-check. It exposes the same observation space, action space and info keys, so every
agent runs on either backend unchanged. All numbers in this document come from the custom
simulator.

## 2. MDP specification

| Element | Definition |
|---|---|
| State / observation | `Box(float32, shape=(6,))`: `[q_N, q_S, q_E, q_W, phase, elapsed]` |
| `q_i` | Vehicles queued on approach `i`, clipped to `[0, 40]` |
| `phase` | Current green phase, `0` (NS) or `1` (EW) |
| `elapsed` | Seconds of green shown in the current phase, clipped to `[0, 120]`. Yellow time does not count. It resets to 0 on a switch, so the first observation after a switch reads 5 |
| Action | `Discrete(2)`: the phase to show next. The same phase as now means hold; the other phase means switch |
| Hold step | 5 s of green in the current phase |
| Switch step | 3 s of yellow, then 5 s of green in the new phase (8 s total) |
| Reward | `r = -0.01 * queue_seconds - 0.1 * switched` |
| `queue_seconds` | Sum of all four queue lengths, taken at the end of every tick in the step (5 or 8 ticks) |
| `switched` | `1` if the step changed phase, else `0` |
| Episode | 3600 simulated seconds. It ends with `truncated = True` and is never `terminated`. That is at most 720 decisions (all holds); each switch makes the step 3 s longer, so a fixed 30/30 timer makes about 655 |
| Discount | `γ = 0.95` per decision |
| Info per step | `sim_time`, `arrivals`, `departures`, `queue_seconds`, `switched` |

The agent decides at `t = 0` and then after every step. The environment code is in
`envs/base_env.py` (spaces, observation builder, info keys) and `envs/simple_env.py`
(dynamics and reward).

### Why this reward

Queue-seconds is the quantity the evaluation metric measures. A vehicle that waits for
`k` seconds contributes `k` queue-seconds, so the total queue-seconds of an episode divided
by its arrivals is the mean delay per vehicle (`avg_wait`, section 5). Maximising the
undiscounted return is the same as minimising total delay plus a small switch cost. The
reward never needs per-vehicle tracking, and the queue lengths that drive it are already
in the observation.

The switch penalty is small: `0.1` equals 10 queue-seconds, or about two cars waiting
through one 5 s step. The real cost of switching is already in the reward. A switch step
lasts 8 s, and the queues keep growing through the 3 s of yellow, so that lost time shows
up as extra queue-seconds. The penalty only discourages flicker between near-equal choices.

`reward_scale = 0.01` keeps per-step rewards near `-0.2` to `-1` at typical queue lengths
(4 to 20 queued vehicles over a 5 s step).
It changes the size of the Q-values, not the policy.

### Discount and truncation

With `γ = 0.95` the effective horizon is about `1 / (1 - γ) = 20` decisions, or roughly
100–160 s of simulated time. That covers a few signal cycles, long enough to see the cost
of starving one direction, and short enough to learn quickly.

The hour limit is a time limit, not a natural end, and time-of-day is not in the state.
So the update bootstraps from `s'` on the last step as well (only `terminated` stops
bootstrapping, and it is always `False`).

## 3. Tabular state

The agent maps each observation to a table index with `Discretizer` in
`agents/q_learning.py`. The environment keeps the raw observation, so the same
environment can feed DQN.

| Component | Edges | Bins | Bin ranges |
|---|---|---|---|
| Queue, per approach (x4) | `[3, 7, 13]` | 4 | `<3`, `3–6`, `7–12`, `>=13` vehicles |
| Phase | none | 2 | NS, EW |
| Elapsed green | `[10, 30]` | 3 | `<10`, `10–29`, `>=30` s |

That gives `4^4 * 2 * 3 = 1536` states and a `1536 x 2` Q-table, about 25 KB as float64.

Why these edges:

- Queue edges are spaced wider as queues grow. Under this demand most queues sit between 0
  and 10, so the fine bins cover the range where decisions are made. Every long queue is
  treated alike: switch towards it.
- Elapsed is observed in steps of 5 s. The bins split it into "just switched" (0 or 5 s,
  where switching back wastes the yellow), "normal green" and "long green" (30 s or more,
  where a long red queue is likely building).
- The edges are configuration values (`configs/q_learning.yaml`), not code constants.

Coverage limits: the bins discard detail. A queue of 13 and a queue of 40 look the same,
and so do 30 s and 120 s of green. The table is also sparse in practice. After 400
training episodes, 990 of the 1536 states (64%) in `results/models/q_learning.npz` have
never been updated, so both of their Q-values are still 0. These are queue patterns the
training trajectories rarely reach: under the learned policy on seed 1000, 99.9% of
observed queues are below 13, so the top queue bin is seldom visited. Section 6 covers
why this matters.

## 4. Algorithm and justification

### Options considered

| Algorithm | Strengths | Weaknesses | Role here |
|---|---|---|---|
| Tabular Q-learning | Simple and fast; every Q-value can be inspected; easy to explain in a short demo | Needs a discretized state; does not scale | Primary method |
| SARSA | Almost no extra code once Q-learning exists; an on-policy comparison | Not new on its own | Ablation |
| DQN (Stable-Baselines3) | Works on the raw continuous observation; generalises to unseen states | More training time; sensitive to hyperparameters; harder to explain fully | Stretch goal |
| PPO / actor-critic | Strong in large problems | Overkill at this size; higher risk of not converging in time | Not used |

### Why tabular Q-learning first

The problem is small. Two actions and six observation values make a table of 1536 states
that trains in under a minute (400 episodes took 37 s). The learned policy can be read
directly: for any queue pattern you can look up whether the agent holds or switches, and
by how much it prefers one over the other. Q-learning is also off-policy. It learns the
value of the greedy policy while it explores with ε-greedy actions, which is the policy
the evaluation runs.

### Update rule

After each transition `(s, a, r, s')`:

```
Q(s, a) <- Q(s, a) + α * (r + γ * max_a' Q(s', a') - Q(s, a))
```

The code is `QLearningAgent.update`. The bootstrap term `max_a' Q(s', a')` lives in
`bootstrap_value`, the only method SARSA overrides. All Q-values start at 0.

| Hyperparameter | Value |
|---|---|
| Learning rate `α` | 0.1 |
| Discount `γ` | 0.95 |
| Episodes | 400 (training episode `k` uses env seed `k`) |
| Exploration | ε-greedy. ε falls linearly from 1.0 to 0.05 over the first 70% of episodes (episodes 0–280), then stays at 0.05 |
| Greedy action | argmax over Q, ties broken uniformly at random |

### SARSA (ablation)

SARSA (`agents/sarsa.py`) uses the same discretizer, hyperparameters and
training loop. Only the target changes. It bootstraps from the action the behaviour policy actually takes next:

```
Q(s, a) <- Q(s, a) + α * (r + γ * Q(s', a') - Q(s, a)),   a' = the ε-greedy action taken in s'
```

The shared episode loop already passes `(s, a, r, s', a')` to the update, so SARSA needs no
loop changes. As an on-policy method it learns the value of the exploring policy, which
tends to be more cautious while ε is high. Comparing the two answers whether the
off-policy target helps on this problem (requirement R3). On this problem it does not
make a measurable difference (section 7).

### DQN (stretch)

DQN replaces the table with a neural network on the raw 6-value observation. It has no
binning, and it generalises: a state never seen in training still gets an estimate from
similar states. Section 6 shows why that matters. It needs torch and more tuning, so it
is an optional extra (`pip install -e ".[dqn]"`) and comes after the tabular methods.

The implementation is `agents/dqn.py`, using Stable-Baselines3. Observations are divided
by the observation-space upper bounds so every input lies in [0, 1]. Training episode `k`
uses env seed `k`, as for the tabular agents. Settings (`configs/dqn.yaml`):

| Hyperparameter | Value |
|---|---|
| Network | MLP, two hidden layers of 64 units |
| Learning rate | `5e-4` |
| Replay buffer / warm-up | 50,000 transitions / 1,000 steps |
| Batch / update frequency | 64 / every 4 steps |
| Target network update | every 2,000 steps |
| Exploration | ε from 1.0 to 0.05 over the first 30% of steps |
| Discount `γ` | 0.99 (the tabular agents use 0.95) |
| Episodes | 400, about 3 minutes on CPU |

These settings worked on the first full run, so DQN was not tuned further.

## 5. Baseline and evaluation protocol

The non-learning baseline is a fixed timer (`baselines/fixed_timer.py`). It shows NS green
for `green_ns` seconds and EW green for `green_ew` seconds, and uses the environment's own
3 s yellow. It switches once elapsed green reaches the target, so green times are
multiples of 5 s. At 30/30, a cycle is `30 + 3 + 30 + 3 = 66` s, giving 109 switches an
hour.

It is reported in two forms:

- Default: 30/30.
- Tuned: the best split from a grid search over 10–60 s in steps of 5, run on training
  seeds only.

Evaluation uses env seeds 1000–1009. No agent trains on them and no baseline is tuned on
them. Each controller runs one greedy episode per seed, and results are reported as
mean ± std over the 10 seeds. The requirement (R5) is that Q-learning beats both the
default and the tuned timer on mean `avg_wait`.

Per-episode metrics (`episode.py`):

| Metric | Definition |
|---|---|
| `return` | Sum of rewards |
| `avg_wait` | Total queue-seconds ÷ total arrivals (s per vehicle). Counts vehicles still queued at the end |
| `avg_queue` | Total queue-seconds ÷ simulated seconds (vehicles) |
| `throughput` | Vehicles that departed |
| `switches` | Number of phase changes |

## 6. Lessons so far

The first greedy evaluation of the trained Q-table scored a mean `avg_wait` of
92.7 ± 240 s over the 10 held-out seeds. One seed, 1007, caused it. On that seed the
policy reached states it had never visited in training. Both Q-values there were still 0,
and `argmax` on a tie returns the first index. Action 0 means "NS green", so whenever the
policy hit an unvisited state during NS green it held, and the next state was usually just
as unvisited. NS stayed green for 483 decisions in a row and the EW queues grew to 327
vehicles.

The fix was to break ties at random. An unvisited state now gets a coin flip, which
switches within a few steps and moves the queues back towards states the table knows.
With that change, seed 1007 scores 12.7 s and the mean over seeds 1000–1009 drops to
12.22 s. A unit test covers the tie-breaking.

The lesson is about generalisation, not the tie-break itself. A tabular agent knows
nothing about a state until it has visited it, and with 64% of states unvisited, any
held-out seed with unusual traffic can push it off the map. Random tie-breaking makes the
failure harmless, but it is still a guess. A function approximator such as DQN would rank
the two actions in an unseen state using what it learned in nearby states. That gap is
the main reason DQN is the stretch goal.

## 7. Results

Mean `avg_wait` over held-out seeds 1000–1009, from `results/evaluation.csv`
(`python -m traffic_rl.evaluate`):

| Controller | avg_wait (s) |
|---|---|
| DQN | 9.35 ± 0.32 |
| SARSA | 11.43 ± 0.47 |
| Q-learning | 12.22 ± 0.51 |
| Fixed timer 20/20, tuned on training seeds | 15.14 ± 1.41 |
| Fixed timer 30/30, default | 17.23 ± 0.81 |

All three learning agents meet R5:

- Q-learning waits 19% less than the tuned timer and 29% less than the default.
- The grid search on training seeds picked 20/20. On the held-out seeds 15/15 does
  slightly better (14.84 s). The tuned number is reported as-is, because choosing the
  split on the test seeds would leak test data into the baseline.

**SARSA vs Q-learning.** The table uses one training run per agent (agent seed 0). To
check whether the gap is real, each agent was retrained with agent seeds 0, 1 and 2:

| Agent | Seed 0 | Seed 1 | Seed 2 | Mean of seeds | Std |
|---|---|---|---|---|---|
| Q-learning | 12.22 | 10.81 | 10.93 | 11.32 | 0.64 |
| SARSA | 11.43 | 11.26 | 12.00 | 11.56 | 0.32 |

The difference between the agents is smaller than the spread between seeds of the same
agent, so on this problem neither TD target is clearly better. Seed 0 happens to be
Q-learning's weakest run, which is why the single-run table favours SARSA. All six runs
beat the tuned timer.

**DQN.** DQN waits 23% less than Q-learning and 38% less than the tuned timer. It also
has the lowest spread across seeds. This fits the generalisation argument in section 6:

- the network is not limited to four queue bins
- it ranks actions in rarely seen states using what it learned in nearby ones

Two caveats limit how far this result goes:

- DQN was trained with one agent seed.
- It uses `γ = 0.99` against 0.95 for the tabular agents, so part of the gap may come
  from the longer horizon rather than the function approximator.

Isolating that would need a tabular run with `γ = 0.99` or a DQN run with `γ = 0.95`.

**Learning.** Average wait per training episode falls from about 24 s (ε ≈ 0.9) to about
11 s for the tabular agents and about 9 s for DQN (`results/plots/learning_curve.png`).

| Agent | 20-episode mean drops below the tuned timer around |
|---|---|
| DQN | episode 50 |
| Q-learning | episode 150 |
| SARSA | episode 175 |

## 8. Limitations

- One intersection. There is no coordination between signals and no upstream platoon
  arrivals.
- One lane per approach, through traffic only. No turning movements, no protected-turn
  phases, no pedestrians.
- The simulator is simple. Queues are point queues with no spatial length, so there is no
  spillback, no start-up lost time and no driver variation beyond the per-tick departure
  probability. Arrivals are independent Poisson processes with a stepwise demand schedule.
- Demand covers one hour with one schedule. The agent has not been tested on demand
  patterns very different from training.
- The tabular state cannot tell long queues apart (13 or more vehicles all share one bin)
  and has no notion of time of day.

The SUMO backend is a partial check on simulator fidelity:

- On seed 1000, fixed timers rank the same in both simulators (20/20 < 30/30 < 40/40).
- Absolute waits are lower in SUMO, because it counts only halted vehicles as queued.
- The trained agents have not been evaluated in SUMO.

Multi-intersection control and richer phase plans are out of scope (see
`REQUIREMENTS.md`).
