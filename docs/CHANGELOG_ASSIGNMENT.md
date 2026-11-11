# Assignment Changelog: Fundamentals of Reinforcement Learning

This document records the exact changes made to complete the **Fundamentals of Reinforcement Learning** assignment, detailing the rationale, approach, and implementation for each item.

---

## 1. Overview of Changes

To satisfy the 10 assignment requirements while respecting all repository standards and pre-push verification checks, the project was systematically augmented with formal theoretical documentation, numerical derivations, an interactive MDP demonstration tool, test coverage, and updated user-facing documentation:

1. **Formal Assignment Documentation (`docs/assignment_rl_fundamentals.md`):**
   - Comprehensive, end-to-end report systematically addressing Items 1 through 10.
   - Formulates the 5-tuple Finite MDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$, Markov property proof, reward derivation aligning with total delay minimization, state-transition diagrams, and decision flowcharts.
   - Includes a step-by-step numerical Bellman calculation for an unbalanced state ($s_0 = 61$).

2. **Dedicated Distinction Document (`docs/mdp_vs_rl_distinction.md`):**
   - Specifically fulfills **Item 9** (*"Clearly distinguish the MDP design and analysis from any proposed future RL implementation"*).
   - Contrasts model-based MDP dynamic programming with model-free RL (Q-learning, SARSA, DQN), discusses the curse of dimensionality, state abstraction limits, and outlines future frontiers (continuous Actor-Critic PPO/SAC, Multi-Agent RL / Dec-POMDP, and Constrained MDPs).

3. **Interactive MDP Walkthrough Prototype (`src/traffic_rl/mdp_demo.py`):**
   - Fulfills **Item 8 & Item 9 (first 9)** (*"Develop a simple prototype/demo to demonstrate the designed MDP and decision-making process"*).
   - Provides an executable demonstration that decomposes observations into discrete queue/phase/elapsed bins, executes environment transitions, computes immediate rewards and penalties, and analytically evaluates the one-step Bellman equation values for candidate actions side-by-side with learned Q-values.
   - Implemented adhering strictly to constraint C9 (zero code comments), 100% type annotations, and Ruff formatting.

4. **Unit Test Suite Expansion (`tests/test_mdp_demo.py`):**
   - Adds 4 comprehensive unit tests verifying state decomposition, Bellman step evaluation, output formatting, and CLI parameter parsing.
   - Increases total test suite count from 43 to 47 passing tests.

5. **Updated Repository Guide (`README.md`):**
   - Added a clear, transparent **"Assignment & MDP Enhancements (Old vs. New)"** section.
   - Explicitly contrasts the preexisting baseline with the new formal MDP components, Bellman value analysis, prototype walkthrough, and links to all assignment documentation in `docs/`.

---

## 2. Approach and Implementation by Requirement

### Item 1: Real-World Sequential Decision-Making Problem
- **Approach:** Evaluated real-world applicability of traffic signal management. Emphasized delayed consequences (prolonged green leading to cross-street queue cascades), the cost of yellow clearance lost time, and non-stationary demand distributions across peak hours.
- **Implementation:** Documented in `docs/assignment_rl_fundamentals.md` (Section 1).

### Item 2: Agent, Environment, States, Actions, Goals, Rewards
- **Approach:** Defined the interaction boundary cleanly. Linked the scalar reward function mathematically to the cumulative vehicle delay (queue-seconds) metric.
- **Implementation:** Formulated in `docs/assignment_rl_fundamentals.md` (Section 2) and mapped directly to `src/traffic_rl/envs/base_env.py` and `src/traffic_rl/envs/simple_env.py`.

### Item 3: Finite Markov Decision Process (MDP) Modeling
- **Approach:** Formally specified the 5-tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$. Derived the discrete 1536-state space ($4^4 \times 2 \times 3$), arrival/departure probability distributions (Poisson arrivals and Binomial saturation departures), and validated the Markov property.
- **Implementation:** Documented in `docs/assignment_rl_fundamentals.md` (Section 3).

### Item 4: Episodes, Returns, Policy, and Value Functions
- **Approach:** Stated mathematical expressions for finite-horizon episodic trajectories ($T = 3600$s), discounted return $G_t$ with $\gamma = 0.95$ ($H_{\text{eff}} \approx 20$ decisions), $\epsilon$-greedy exploration vs greedy evaluation policies, and the value functions $V^\pi, Q^\pi, V^*, Q^*$.
- **Implementation:** Documented in `docs/assignment_rl_fundamentals.md` (Section 4).

### Item 5: Visual Representations (Diagrams, Tables, Flowcharts)
- **Approach:** Used GitHub Flavored Markdown and Mermaid diagrams to visualize the MDP structure.
- **Implementation:** 
  - State-transition diagram showing yellow clearance transitions and phase loops (`docs/assignment_rl_fundamentals.md`, Section 5.1).
  - Decision epoch interaction flowchart (`docs/assignment_rl_fundamentals.md`, Section 5.2).
  - State discretization binning table, action mapping table, and demand schedules (Sections 5.3 and 5.4).

### Item 6: Bellman Equation Application and State Value Calculation
- **Approach:** Formulated the Bellman Expectation and Optimality Equations. Selected a concrete, physically intuitive unbalanced state $s_0 = (q_N=2, q_S=2, q_E=8, q_W=8, p=0, \tau=15\text{s})$ (state index 61).
- **Implementation:** 
  - Performed step-by-step analytical and numerical calculations for Action 0 (HOLD) vs. Action 1 (SWITCH).
  - Calculated expected immediate reward $\mathbb{E}[R \mid s_0, a]$, discounted continuation value $\gamma \mathbb{E}[V^*(S')]$, and Bellman RHS values.
  - Proved mathematically why $Q^*(s_0, 1) = -4.714 > Q^*(s_0, 0) = -6.211$, commanding a SWITCH action (`docs/assignment_rl_fundamentals.md`, Section 6).

### Item 7: Explanation of Policy Influence on Decisions
- **Approach:** Analyzed the empirical effect of $\epsilon$-greedy exploration decay, the critical role of random tie-breaking in avoiding infinite green lockups on unvisited states (seed 1007 discovery), and dynamic phase split adaptation across demand shifts.
- **Implementation:** Documented in `docs/assignment_rl_fundamentals.md` (Section 7).

### Item 8 & 9 (First 9): Prototype / Demonstration of the MDP
- **Approach:** Provided both an interactive live web application and a dedicated lightweight CLI prototype module.
- **Implementation:**
  - Authored `src/traffic_rl/mdp_demo.py` featuring state decomposition, transition execution, and one-step Bellman evaluation.
  - Authored unit test suite in `tests/test_mdp_demo.py`.
  - Maintained live deployment on Streamlit Community Cloud (`app/streamlit_app.py`).

### Item 9 (Second 9): Distinction between MDP Design and Future RL Implementations
- **Approach:** Created a dedicated, standalone markdown document specifically isolating the theoretical MDP boundary from algorithmic reinforcement learning and future extensions.
- **Implementation:** Dedicated file authored at `docs/mdp_vs_rl_distinction.md`, cross-referenced in `docs/assignment_rl_fundamentals.md` and `README.md`.

### Item 10: References, Examples, and Empirical Evidence
- **Approach:** Supported all formulations with citations to foundational literature (Sutton & Barto, Watkins, Mnih, HCM 2016) and empirical benchmarking tables comparing Q-learning, SARSA, and DQN against default and tuned fixed-time baselines across 10 held-out seeds.
- **Implementation:** Documented in `docs/assignment_rl_fundamentals.md` (Section 10) and `docs/references.md`.

---

## 3. Verification and Integrity Check

Before pushing, the changes were verified using the project's strict check harness:
- `ruff check .`: Passed (0 errors).
- `ruff format --check .`: Passed (31 files formatted).
- `mypy`: Passed (0 errors in 19 source files).
- `pytest -q`: Passed (47 passed in 15.4s).
- Smoke training: Passed (5 episodes trained cleanly).
- Pre-push hook: Verified and passing.
