"""
AGENT02: Reinforcement-Learning-Based Agent vs Rule-Based Agent
================================================================

Course: Intelligent System Development
Laboratory Experiment 02

Run (from any directory):
    python agent02_experiment.py

Dependencies:
    numpy, matplotlib

Outputs (written to ../results/ next to this file):
    *.csv                 raw and summarised measurements
    q_tables.txt          learned Q-tables (baseline, RL vs RL, modified market)
    figures/*.png         figures used in the report
    tables/*.tex          LaTeX tables generated from the measurements

This file starts from the reference implementation supplied with the
assignment (agent02_code.tex). The environment, the Q-learning update and the
rule-based policy are unchanged. Changes, all documented in the report:

  1. Every agent exposes act(state, explore) and learn(...); non-learning
     agents ignore learn(). This lets the same training / evaluation loop run
     RL vs Rule, Rule vs Rule and RL vs RL.
  2. Fair comparison (Experiment E): the rule-based agent is evaluated in the
     same seat as the RL agent (seat 0), against the same opponent, with the
     same environment seed. The environment draws the same number of random
     numbers per step whatever the actions are, so both runs see exactly the
     same demand sequence (common random numbers).
  3. Every configuration is repeated over 10 independent seeds and reported
     as mean and standard deviation across seeds.
  4. Exploration schedules include fixed epsilon (0.05, 0.30) and a purely
     greedy agent (epsilon = 0), as suggested in Section 7.4.
  5. Extra experiments for Questions 8-10: two simultaneously learning
     Q-agents, a modified market with an aggressive competitor, and a
     robustness test of an improved agent against a pool of opponents.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class Config:
    episodes: int = 5000
    steps_per_episode: int = 20

    alpha: float = 0.10
    gamma: float = 0.90

    epsilon_start: float = 0.20
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.995

    price: float = 10.0
    unit_cost: float = 5.0
    discount_rate: float = 0.20

    low_demand_customers: int = 10
    high_demand_customers: int = 20

    seed: int = 42
    evaluation_episodes: int = 500

    # Window K (in episodes) for the learning curves of Experiment B.
    window: int = 100


SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
EVAL_SEED_OFFSET = 800_000

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
FIG = OUT / "figures"
TAB = OUT / "tables"

State = tuple[int, int]
NO_DISCOUNT, DISCOUNT = 0, 1


# ============================================================
# 2. E-COMMERCE ENVIRONMENT  (unchanged from the reference)
# ============================================================

class ECommerceEnvironment:
    """
    State for agent i: (demand, competitor_previous_action)
    Action: 0 = no discount, 1 = discount
    Reward: profit = sales * (effective_price - unit_cost)
    """

    def __init__(self, config, seed=None):
        self.config = config
        actual_seed = config.seed if seed is None else seed
        self.rng = np.random.default_rng(actual_seed)
        self.demand = 0
        self.previous_actions = [0, 0]

    def reset(self):
        self.demand = int(self.rng.integers(0, 2))
        self.previous_actions = [0, 0]
        return self.get_states()

    def get_states(self):
        return [
            (self.demand, self.previous_actions[1]),
            (self.demand, self.previous_actions[0]),
        ]

    def step(self, actions):
        if len(actions) != 2:
            raise ValueError("Exactly two actions are required.")
        if any(action not in (0, 1) for action in actions):
            raise ValueError("Actions must be 0 or 1.")

        customers = (self.config.low_demand_customers if self.demand == 0
                     else self.config.high_demand_customers)

        rewards, sales_list, conversion_list = [], [], []
        for i in range(2):
            own_action = actions[i]
            competitor_action = actions[1 - i]

            conversion = (0.30 + 0.25 * self.demand
                          + 0.15 * own_action - 0.10 * competitor_action)
            conversion = float(np.clip(conversion, 0.05, 0.95))

            sales = int(self.rng.binomial(customers, conversion))
            effective_price = self.config.price * (
                1.0 - self.config.discount_rate * own_action)
            profit = sales * (effective_price - self.config.unit_cost)

            rewards.append(float(profit))
            sales_list.append(sales)
            conversion_list.append(conversion)

        self.previous_actions = list(actions)
        self.demand = int(self.rng.integers(0, 2))

        info = {"sales": sales_list, "conversion": conversion_list,
                "customers": customers}
        return self.get_states(), rewards, info


def expected_profit(config, demand, own_action, competitor_action):
    """Expected one-step profit E[r] = N * p * margin (used for analysis only)."""
    customers = (config.low_demand_customers if demand == 0
                 else config.high_demand_customers)
    p = np.clip(0.30 + 0.25 * demand + 0.15 * own_action
                - 0.10 * competitor_action, 0.05, 0.95)
    margin = config.price * (1 - config.discount_rate * own_action) - config.unit_cost
    return customers * p * margin


# ============================================================
# 3. AGENTS
# ============================================================

class QLearningAgent:
    """Tabular Q-learning agent, Q[demand, competitor_previous_action, action].

    step_size="constant" is the reference agent (fixed alpha).
    step_size="visit" uses alpha = 1 / N(s, a) (sample average), which is the
    learning rule of the improved agent in Question 10.
    """

    is_learner = True

    def __init__(self, alpha=0.10, gamma=0.90, epsilon=0.20, seed=123,
                 step_size="constant"):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.step_size = step_size
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((2, 2, 2), dtype=float)
        self.visits = np.zeros((2, 2, 2), dtype=int)

    def act(self, state, explore=True):
        demand, competitor_previous_action = state
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 2))
        values = self.q[demand, competitor_previous_action]
        best_actions = np.flatnonzero(np.isclose(values, np.max(values)))
        return int(self.rng.choice(best_actions))

    def learn(self, state, action, reward, next_state, terminal=False):
        d, c = state
        nd, nc = next_state
        old_value = self.q[d, c, action]
        target = reward if terminal else reward + self.gamma * np.max(self.q[nd, nc])

        self.visits[d, c, action] += 1
        if self.step_size == "visit":
            alpha = 1.0 / self.visits[d, c, action]
        else:
            alpha = self.alpha
        self.q[d, c, action] = old_value + alpha * (target - old_value)

    def reset(self):
        """Called at the start of every episode (the reference agent has no episode memory)."""

    def greedy_policy(self):
        return {(d, c): int(np.argmax(self.q[d, c]))
                for d in range(2) for c in range(2)}

    def policy_code(self):
        """Greedy policy over the 4 states packed into one integer (for flip counting)."""
        p = self.greedy_policy()
        return p[(0, 0)] + 2 * p[(0, 1)] + 4 * p[(1, 0)] + 8 * p[(1, 1)]


class MemoryQAgent(QLearningAgent):
    """Improved agent for Question 10.

    State = (demand, competitor_previous_action, own_previous_action).
    The reference rule reacts to *our* previous action, so the effect of a
    discount shows up one step later. With the reference state (d, c) that
    consequence is invisible to the learner; adding the agent's own previous
    action makes it part of the state. Q has 2 x 2 x 2 x 2 = 16 entries.
    """

    def __init__(self, gamma=0.90, epsilon=0.20, seed=123, step_size="visit", alpha=0.10):
        super().__init__(alpha, gamma, epsilon, seed, step_size)
        self.q = np.zeros((2, 2, 2, 2), dtype=float)
        self.visits = np.zeros((2, 2, 2, 2), dtype=int)
        self.reset()

    def reset(self):
        self.own_prev = 0
        self._last_state = (0, 0, 0)

    def act(self, state, explore=True):
        s = (state[0], state[1], self.own_prev)
        if explore and self.rng.random() < self.epsilon:
            a = int(self.rng.integers(0, 2))
        else:
            values = self.q[s]
            a = int(self.rng.choice(np.flatnonzero(np.isclose(values, values.max()))))
        self._last_state = s
        self.own_prev = a
        return a

    def learn(self, state, action, reward, next_state, terminal=False):
        s = self._last_state
        ns = (next_state[0], next_state[1], action)
        target = reward if terminal else reward + self.gamma * np.max(self.q[ns])
        self.visits[s + (action,)] += 1
        alpha = (1.0 / self.visits[s + (action,)] if self.step_size == "visit"
                 else self.alpha)
        self.q[s + (action,)] += alpha * (target - self.q[s + (action,)])

    def greedy_policy(self):
        return {(d, c, p): int(np.argmax(self.q[d, c, p]))
                for d in range(2) for c in range(2) for p in range(2)}

    def policy_code(self):
        return sum(a << i for i, a in enumerate(self.greedy_policy().values()))


class RuleBasedAgent:
    """Reference rule: discount only when demand is high and the competitor
    discounted in the previous step."""

    is_learner = False
    name = "Rule-based"

    def act(self, state, explore=True):
        demand, competitor_previous_action = state
        if demand == 0:
            return NO_DISCOUNT
        elif demand == 1 and competitor_previous_action == 1:
            return DISCOUNT
        else:
            return NO_DISCOUNT

    def learn(self, *args, **kwargs):
        pass

    def reset(self):
        pass


class AggressiveAgent(RuleBasedAgent):
    """Modified competitor (Question 9): discounts whenever demand is high."""

    name = "Aggressive"

    def act(self, state, explore=True):
        return DISCOUNT if state[0] == 1 else NO_DISCOUNT


class TitForTatAgent(RuleBasedAgent):
    """Copies the competitor's previous action."""

    name = "Tit-for-tat"

    def act(self, state, explore=True):
        return state[1]


class FixedAgent(RuleBasedAgent):
    def __init__(self, action, name):
        self.action = action
        self.name = name

    def act(self, state, explore=True):
        return self.action


class RandomAgent(RuleBasedAgent):
    name = "Random (p=0.5)"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, state, explore=True):
        return int(self.rng.integers(0, 2))


def make_opponent(kind, seed=0):
    return {
        "rule": RuleBasedAgent,
        "aggressive": AggressiveAgent,
        "tit_for_tat": TitForTatAgent,
        "always_discount": lambda: FixedAgent(DISCOUNT, "Always discount"),
        "never_discount": lambda: FixedAgent(NO_DISCOUNT, "Never discount"),
        "random": lambda: RandomAgent(seed + 30_000),
    }[kind]()


# ============================================================
# 4. TRAINING
# ============================================================

def epsilon_at(episode, schedule):
    eps_start, eps_min, eps_decay = schedule
    return max(eps_min, eps_start * eps_decay ** episode)


def train(config, seed, alpha=None, schedule=None, opponent="rule",
          step_size="constant", opponent_pool=None, memory=False):
    """Train the agent in seat 0 for config.episodes episodes.

    opponent: "rule", "aggressive", ... or "rl" (a second, independent
              Q-learner with its own RNG that learns at the same time).
    opponent_pool: list of opponent kinds; one is drawn at random for every
              episode (used by the improved agent of Question 10).

    Returns the seat-0 agent, the seat-1 agent and one row per episode.
    """
    alpha = config.alpha if alpha is None else alpha
    if schedule is None:
        schedule = (config.epsilon_start, config.epsilon_min, config.epsilon_decay)

    env = ECommerceEnvironment(config, seed=seed)
    if memory:
        agent = MemoryQAgent(config.gamma, schedule[0], seed + 10_000, step_size, alpha)
    else:
        agent = QLearningAgent(alpha, config.gamma, schedule[0], seed + 10_000,
                               step_size=step_size)

    pool_rng = np.random.default_rng(seed + 40_000)
    if opponent_pool:
        pool = [make_opponent(k, seed) for k in opponent_pool]
        other = pool[0]
    elif opponent == "rl":
        other = QLearningAgent(alpha, config.gamma, schedule[0], seed + 20_000)
    else:
        other = make_opponent(opponent, seed)

    rows = []
    T = config.steps_per_episode
    for episode in range(config.episodes):
        eps = epsilon_at(episode, schedule)
        agent.epsilon = eps
        if other.is_learner:
            other.epsilon = eps
        if opponent_pool:
            other = pool[int(pool_rng.integers(0, len(pool)))]

        states = env.reset()
        agent.reset()
        other.reset()
        totals = [0.0, 0.0]
        discounts = [0, 0]
        for step in range(T):
            actions = [agent.act(states[0], explore=True),
                       other.act(states[1], explore=True)]
            next_states, rewards, _ = env.step(actions)
            terminal = step == T - 1
            agent.learn(states[0], actions[0], rewards[0], next_states[0], terminal)
            other.learn(states[1], actions[1], rewards[1], next_states[1], terminal)
            for i in range(2):
                totals[i] += rewards[i]
                discounts[i] += actions[i]
            states = next_states

        rows.append({
            "episode": episode + 1,
            "rl_episode_reward": totals[0],
            "opponent_episode_reward": totals[1],
            "rl_mean_reward_per_step": totals[0] / T,
            "opponent_mean_reward_per_step": totals[1] / T,
            "rl_discount_rate": discounts[0] / T,
            "opponent_discount_rate": discounts[1] / T,
            "epsilon": eps,
            "alpha": alpha,
            "rl_policy": agent.policy_code(),
            "opponent_policy": other.policy_code() if other.is_learner else -1,
        })
    return agent, other, rows


# ============================================================
# 5. SEPARATE POLICY EVALUATION
# ============================================================

def evaluate(config, agent0, agent1, seed, episodes=None):
    """Evaluate two frozen policies against each other.

    No learning happens and exploration is disabled (epsilon = 0) for both
    seats. The evaluation environment uses its own seed, different from any
    training seed.
    """
    episodes = config.evaluation_episodes if episodes is None else episodes
    env = ECommerceEnvironment(config, seed=seed)
    agents = [agent0, agent1]
    T = config.steps_per_episode

    total = np.zeros(2)
    counts = np.zeros((2, 2), dtype=int)
    episode_rewards = np.zeros((episodes, 2))
    for ep in range(episodes):
        states = env.reset()
        for a in agents:
            a.reset()
        for _ in range(T):
            actions = [agents[i].act(states[i], explore=False) for i in range(2)]
            states, rewards, _ = env.step(actions)
            for i in range(2):
                total[i] += rewards[i]
                episode_rewards[ep, i] += rewards[i]
                counts[i, actions[i]] += 1

    steps = episodes * T
    return [{
        "seat": i,
        "total_reward": float(total[i]),
        "mean_reward_per_step": float(total[i] / steps),
        "episode_std_of_mean_reward": float(np.std(episode_rewards[:, i] / T, ddof=1)),
        "no_discount_actions": int(counts[i, 0]),
        "discount_actions": int(counts[i, 1]),
        "discount_action_rate": float(counts[i, 1] / steps),
    } for i in range(2)], episode_rewards / T


# ============================================================
# 6. OUTPUT HELPERS
# ============================================================

def write_csv(filepath, rows):
    if not rows:
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def q_table_text(title, agent):
    lines = [title, "Q[demand, competitor_previous_action, action];"
             " action 0 = no discount, 1 = discount"]
    for d in range(2):
        for c in range(2):
            q0, q1 = agent.q[d, c]
            lines.append(
                f"  state=({d}, {c}) | Q0={q0:9.4f} | Q1={q1:9.4f} | "
                f"visits=({agent.visits[d, c, 0]}, {agent.visits[d, c, 1]}) | "
                f"greedy={'DISCOUNT' if q1 > q0 else 'NO DISCOUNT'}")
    return "\n".join(lines) + "\n\n"


def tex_table(path, header, rows, align):
    body = [r"\begin{tabular}{" + align + "}", r"\toprule",
            " & ".join(header) + r" \\", r"\midrule"]
    body += [" & ".join(str(x) for x in r) + r" \\" for r in rows]
    body += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def windowed(values, k):
    """Mean over consecutive, non-overlapping windows of k episodes (Exp. B)."""
    v = np.asarray(values, dtype=float)
    n = len(v) // k
    return v[: n * k].reshape(n, k).mean(axis=1)


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def policy_flips(codes):
    codes = np.asarray(codes)
    return int(np.count_nonzero(codes[1:] != codes[:-1]))


def is_never_discount(agent):
    return all(a == NO_DISCOUNT for a in agent.greedy_policy().values())


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})
COLORS = ["#2a6fdb", "#e0782f", "#2a9d6f", "#c0392b", "#7d5ba6"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


# ============================================================
# 7. ONE CONFIGURATION OVER MANY SEEDS
# ============================================================

def run_config(config, label, seeds=SEEDS, **train_kwargs):
    """Train + evaluate one configuration over all seeds.

    For every seed the trained RL agent (seat 0) is evaluated greedily against
    the training opponent with seed + EVAL_SEED_OFFSET. Returns per-seed rows,
    the stacked windowed training curves and the trained agents.
    """
    opponent = train_kwargs.get("opponent", "rule")
    per_seed, curves, agents = [], [], []
    for seed in seeds:
        agent, other, rows = train(config, seed, **train_kwargs)
        eval_opp = make_opponent(opponent if opponent != "rl" else "rule", seed)
        summary, _ = evaluate(config, agent, eval_opp, seed + EVAL_SEED_OFFSET)
        train_reward = np.mean([r["rl_mean_reward_per_step"] for r in rows])
        last_reward = np.mean([r["rl_mean_reward_per_step"] for r in rows[-500:]])
        per_seed.append({
            "configuration": label,
            "seed": seed,
            "train_mean_reward_per_step": train_reward,
            "train_last500_mean_reward_per_step": last_reward,
            "eval_mean_reward_per_step": summary[0]["mean_reward_per_step"],
            "eval_discount_rate": summary[0]["discount_action_rate"],
            "never_discount_policy": int(is_never_discount(agent)),
            "policy_flips_during_training": policy_flips([r["rl_policy"] for r in rows]),
        })
        curves.append(windowed([r["rl_mean_reward_per_step"] for r in rows], config.window))
        agents.append(agent)
    return per_seed, np.array(curves), agents


def summarise(label, per_seed):
    ev_m, ev_s = mean_std([r["eval_mean_reward_per_step"] for r in per_seed])
    tr_m, tr_s = mean_std([r["train_mean_reward_per_step"] for r in per_seed])
    dr_m, _ = mean_std([r["eval_discount_rate"] for r in per_seed])
    return {
        "configuration": label,
        "n_seeds": len(per_seed),
        "eval_mean_reward": ev_m, "eval_std_across_seeds": ev_s,
        "train_mean_reward": tr_m, "train_std_across_seeds": tr_s,
        "eval_discount_rate": dr_m,
        "seeds_with_never_discount_policy": sum(r["never_discount_policy"] for r in per_seed),
        "mean_policy_flips": float(np.mean([r["policy_flips_during_training"] for r in per_seed])),
    }


def rule_in_seat0(config, opponent="rule", seeds=SEEDS):
    """Rule-based agent in seat 0, same opponent and evaluation seeds as the RL agent."""
    rows = []
    for seed in seeds:
        summary, _ = evaluate(config, RuleBasedAgent(), make_opponent(opponent, seed),
                              seed + EVAL_SEED_OFFSET)
        rows.append({
            "configuration": "Rule-based",
            "seed": seed,
            "train_mean_reward_per_step": float("nan"),
            "train_last500_mean_reward_per_step": float("nan"),
            "eval_mean_reward_per_step": summary[0]["mean_reward_per_step"],
            "eval_discount_rate": summary[0]["discount_action_rate"],
            "never_discount_policy": 0,
            "policy_flips_during_training": 0,
        })
    return rows


# ============================================================
# 8. EXPERIMENTS
# ============================================================

def experiment_model(config):
    print("\n== Expected one-step profit (analysis of the model) ==")
    rows = []
    for d in range(2):
        for c in range(2):
            row = {"demand": d, "competitor_current_action": c}
            for a in range(2):
                row[f"E_profit_action_{a}"] = expected_profit(config, d, a, c)
            row["myopic_best_action"] = int(row["E_profit_action_1"] > row["E_profit_action_0"])
            rows.append(row)
            print(row)
    write_csv(OUT / "expected_profit_table.csv", rows)
    tex_table(TAB / "expected_profit.tex",
              ["Demand $d$", "Rival's current action", r"$E[r\mid a=0]$",
               r"$E[r\mid a=1]$", "Better action"],
              [["low" if r["demand"] == 0 else "high",
                "discount" if r["competitor_current_action"] else "no discount",
                f2(r["E_profit_action_0"]), f2(r["E_profit_action_1"]),
                "discount" if r["myopic_best_action"] else "no discount"] for r in rows],
              "llrrl")


def experiment_a_b(config):
    print("\n== Experiment A/B: baseline (seed 42) ==")
    agent, _, rows = train(config, config.seed)
    write_csv(OUT / "baseline_episode_rewards.csv", rows)

    eval_seed = config.seed + 500_000
    rl_summary, rl_eps = evaluate(config, agent, RuleBasedAgent(), eval_seed)
    rule_summary, rule_eps = evaluate(config, RuleBasedAgent(), RuleBasedAgent(), eval_seed)

    summary_rows = []
    for match, summ in [("RL vs Rule", rl_summary), ("Rule vs Rule", rule_summary)]:
        for s in summ:
            agent_name = ("RL" if (match == "RL vs Rule" and s["seat"] == 0)
                          else "Rule-based")
            summary_rows.append({"match": match, "agent": agent_name, **s})
    write_csv(OUT / "baseline_evaluation.csv", summary_rows)
    write_csv(OUT / "baseline_evaluation_episodes.csv", [
        {"episode": i + 1, "rl_seat0_vs_rule": rl_eps[i, 0], "rule_seat1": rl_eps[i, 1],
         "rule_seat0_vs_rule": rule_eps[i, 0]} for i in range(len(rl_eps))])

    with open(OUT / "q_tables.txt", "w", encoding="utf-8") as f:
        f.write(q_table_text("Baseline RL agent (seed 42), trained against the rule-based agent",
                             agent))

    for r in summary_rows:
        print(f"{r['match']:13s} seat {r['seat']} {r['agent']:10s} "
              f"total={r['total_reward']:.0f} mean/step={r['mean_reward_per_step']:.3f} "
              f"discounts={r['discount_actions']} no-discounts={r['no_discount_actions']}")
    print(q_table_text("Baseline Q-table", agent))

    # Training-phase totals (Experiment A asks for total reward of each agent).
    train_totals = {
        "rl_total_training_reward": sum(r["rl_episode_reward"] for r in rows),
        "rule_total_training_reward": sum(r["opponent_episode_reward"] for r in rows),
        "rl_training_discount_actions": int(round(sum(r["rl_discount_rate"] for r in rows)
                                                 * config.steps_per_episode)),
        "rule_training_discount_actions": int(round(sum(r["opponent_discount_rate"] for r in rows)
                                                   * config.steps_per_episode)),
        "total_training_steps": config.episodes * config.steps_per_episode,
        "rl_policy_flips": policy_flips([r["rl_policy"] for r in rows]),
    }
    write_csv(OUT / "baseline_training_totals.csv", [train_totals])
    print(train_totals)

    # Tables for the report.
    tex_table(TAB / "baseline_eval.tex",
              ["Match (seat 0 vs seat 1)", "Agent", "Total reward", "Mean reward/step",
               "Discounts", "No discounts", "Discount rate"],
              [[r["match"], f"{r['agent']} (seat {r['seat']})", f"{r['total_reward']:,.0f}",
                f3(r["mean_reward_per_step"]), r["discount_actions"],
                r["no_discount_actions"], f3(r["discount_action_rate"])]
               for r in summary_rows], "llrrrrr")
    tex_table(TAB / "baseline_qtable.tex",
              ["State $(d, c)$", "$Q(s, 0)$", "$Q(s, 1)$", "Visits $(a{=}0, a{=}1)$",
               "Greedy action"],
              [[f"({d}, {c})", f2(agent.q[d, c, 0]), f2(agent.q[d, c, 1]),
                f"({agent.visits[d, c, 0]:,}, {agent.visits[d, c, 1]:,})",
                "discount" if agent.q[d, c, 1] > agent.q[d, c, 0] else "no discount"]
               for d in range(2) for c in range(2)], "lrrrl")

    # Figure: Experiment B learning curve.
    k = config.window
    rl_w = windowed([r["rl_mean_reward_per_step"] for r in rows], k)
    op_w = windowed([r["opponent_mean_reward_per_step"] for r in rows], k)
    disc_w = windowed([r["rl_discount_rate"] for r in rows], k)
    eps_w = windowed([r["epsilon"] for r in rows], k)
    x = np.arange(1, len(rl_w) + 1) * k

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ep = np.arange(1, len(rows) + 1)
    ax.plot(ep, [r["rl_mean_reward_per_step"] for r in rows], color=COLORS[0], alpha=0.12, lw=0.6)
    ax.plot(x, rl_w, color=COLORS[0], lw=2, marker="o", ms=3, label=f"RL agent (window K={k})")
    ax.plot(x, op_w, color=COLORS[1], lw=2, marker="s", ms=3, label=f"Rule-based agent (K={k})")
    ax.axhline(35.0, color="gray", ls="--", lw=1, label="35 = expected reward if nobody discounts")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean reward per step")
    ax.set_title("Training reward (faint line: single episodes)")
    ax.set_ylim(20, 50)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    ax.plot(x, disc_w, color=COLORS[0], lw=2, label="RL discount rate")
    ax.plot(x, eps_w, color=COLORS[3], lw=1.5, ls="--", label="exploration rate ε")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Rate")
    ax.set_title("RL discount rate and ε per window")
    ax.legend(fontsize=8)
    save(fig, "baseline_training.png")

    # Figure: evaluation episodes RL vs rule (same demand sequence).
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(min(rl_eps[:, 0].min(), rule_eps[:, 0].min()),
                       max(rl_eps[:, 0].max(), rule_eps[:, 0].max()), 30)
    ax.hist(rule_eps[:, 0], bins=bins, alpha=0.55, color=COLORS[1], label="Rule-based (seat 0)")
    ax.hist(rl_eps[:, 0], bins=bins, alpha=0.55, color=COLORS[0], label="RL (seat 0)")
    ax.set_xlabel("Mean reward per step in one evaluation episode")
    ax.set_ylabel("Episodes")
    ax.set_title("Evaluation: 500 episodes, same demand sequence")
    ax.legend(fontsize=8)
    save(fig, "baseline_eval_hist.png")
    return agent


def experiment_c(config):
    print("\n== Experiment C: learning rate ==")
    results, curves = {}, {}
    for alpha in [0.01, 0.10, 0.50]:
        label = f"alpha={alpha:.2f}"
        per_seed, c, agents = run_config(config, label, alpha=alpha)
        results[alpha] = per_seed
        curves[alpha] = c
        s = summarise(label, per_seed)
        print(s)
        if alpha == 0.50:
            with open(OUT / "q_tables.txt", "a", encoding="utf-8") as f:
                f.write(q_table_text(f"alpha=0.50, seed {SEEDS[0]}", agents[0]))
    rows = [r for a in results for r in results[a]]
    write_csv(OUT / "learning_rate_per_seed.csv", rows)
    write_csv(OUT / "learning_rate_comparison.csv",
              [{"alpha": a, **summarise(f"alpha={a:.2f}", results[a])} for a in results])

    k = config.window
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, a in enumerate(curves):
        m, s = curves[a].mean(axis=0), curves[a].std(axis=0, ddof=1)
        x = np.arange(1, len(m) + 1) * k
        axes[0].plot(x, m, color=COLORS[i], lw=2, label=f"α = {a}")
        axes[0].fill_between(x, m - s, m + s, color=COLORS[i], alpha=0.15)
    axes[0].axhline(35.0, color="gray", ls="--", lw=1)
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel(f"Mean reward per step (window K={k})")
    axes[0].set_title("Training curves, mean ± 1 std over 10 seeds")
    axes[0].legend(fontsize=8)

    labels = [f"α = {a}" for a in results]
    means = [np.mean([r["eval_mean_reward_per_step"] for r in results[a]]) for a in results]
    stds = [np.std([r["eval_mean_reward_per_step"] for r in results[a]], ddof=1) for a in results]
    axes[1].bar(labels, means, yerr=stds, capsize=6, color=COLORS[:3], alpha=0.85)
    for j, a in enumerate(results):
        axes[1].scatter([j] * len(results[a]), [r["eval_mean_reward_per_step"] for r in results[a]],
                        color="black", s=10, zorder=3)
    axes[1].set_ylim(min(means) - 3, max(means) + 2)
    axes[1].set_ylabel("Evaluation mean reward per step")
    axes[1].set_title("Greedy evaluation (dots = seeds)")
    save(fig, "learning_rate_comparison.png")
    return results


EXPLORATION = [
    ("Reference decay 0.20→0.02", (0.20, 0.02, 0.995)),
    ("Fixed ε = 0.05", (0.05, 0.05, 1.0)),
    ("Fixed ε = 0.30", (0.30, 0.30, 1.0)),
    ("Greedy, ε = 0", (0.0, 0.0, 1.0)),
]


def experiment_d(config):
    print("\n== Experiment D: exploration ==")
    results, curves = {}, {}
    for label, sched in EXPLORATION:
        per_seed, c, _ = run_config(config, label, schedule=sched)
        results[label] = per_seed
        curves[label] = c
        print(summarise(label, per_seed))
    write_csv(OUT / "exploration_per_seed.csv", [r for l in results for r in results[l]])
    write_csv(OUT / "exploration_comparison.csv",
              [{"epsilon_start": s[0], "epsilon_min": s[1], "epsilon_decay": s[2],
                **summarise(l, results[l])} for l, s in EXPLORATION])

    k = config.window
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (label, _) in enumerate(EXPLORATION):
        m = curves[label].mean(axis=0)
        x = np.arange(1, len(m) + 1) * k
        axes[0].plot(x, m, color=COLORS[i], lw=2, label=label)
    axes[0].axhline(35.0, color="gray", ls="--", lw=1)
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel(f"Training reward per step (window K={k})")
    axes[0].set_title("Training curves (mean over 10 seeds)")
    axes[0].legend(fontsize=8)

    w = 0.38
    xs = np.arange(len(EXPLORATION))
    tr = [summarise(l, results[l]) for l, _ in EXPLORATION]
    axes[1].bar(xs - w / 2, [t["train_mean_reward"] for t in tr], w,
                yerr=[t["train_std_across_seeds"] for t in tr], capsize=4,
                color=COLORS[1], alpha=0.85, label="training (all episodes)")
    axes[1].bar(xs + w / 2, [t["eval_mean_reward"] for t in tr], w,
                yerr=[t["eval_std_across_seeds"] for t in tr], capsize=4,
                color=COLORS[0], alpha=0.85, label="greedy evaluation")
    axes[1].set_xticks(xs, [l.replace(", ", "\n").replace(" 0.20", "\n0.20")
                            for l, _ in EXPLORATION], fontsize=8)
    axes[1].set_ylim(25, 38)
    axes[1].set_ylabel("Mean reward per step")
    axes[1].set_title("Training vs evaluation reward (± std over seeds)")
    axes[1].legend(fontsize=8, loc="lower left")
    save(fig, "exploration_comparison.png")
    return results


def experiment_e(config, lr_results, exp_results):
    print("\n== Experiment E: fair comparison, 10 seeds ==")
    rule_rows = rule_in_seat0(config)
    write_csv(OUT / "rule_based_per_seed.csv", rule_rows)

    table = [
        ("RL, baseline ($\\alpha=0.10$, decaying $\\varepsilon$)", lr_results[0.10]),
        ("RL, low learning rate ($\\alpha=0.01$)", lr_results[0.01]),
        ("RL, high learning rate ($\\alpha=0.50$)", lr_results[0.50]),
        ("RL, fixed $\\varepsilon=0.05$", exp_results["Fixed ε = 0.05"]),
        ("RL, fixed $\\varepsilon=0.30$", exp_results["Fixed ε = 0.30"]),
        ("RL, greedy $\\varepsilon=0$", exp_results["Greedy, ε = 0"]),
        ("Rule-based", rule_rows),
    ]
    out_rows, tex_rows = [], []
    for label, rows in table:
        s = summarise(label, rows)
        out_rows.append(s)
        tex_rows.append([label, f3(s["eval_mean_reward"]), f3(s["eval_std_across_seeds"]),
                         f3(s["eval_discount_rate"]),
                         "--" if label == "Rule-based"
                         else f"{s['seeds_with_never_discount_policy']}/10"])
        print(f"{label:55s} mean={s['eval_mean_reward']:.3f} std={s['eval_std_across_seeds']:.3f} "
              f"disc={s['eval_discount_rate']:.3f}")
    write_csv(OUT / "multi_seed_summary.csv", out_rows)
    tex_table(TAB / "results_table.tex",
              ["Agent / configuration", "Mean reward", "Std.\\ dev.", "Discount rate",
               "``Never discount'' policy"], tex_rows, "lrrrc")

    # Paired difference RL baseline - rule-based (same seeds, same demand).
    diffs = [a["eval_mean_reward_per_step"] - b["eval_mean_reward_per_step"]
             for a, b in zip(lr_results[0.10], rule_rows)]
    m, s = mean_std(diffs)
    print(f"Paired difference RL - Rule: mean={m:.3f}, std={s:.3f}")
    write_csv(OUT / "paired_difference_rl_minus_rule.csv",
              [{"seed": sd, "difference": d} for sd, d in zip(SEEDS, diffs)])

    # Tables for learning rate and exploration (training + evaluation).
    tex_table(TAB / "learning_rate.tex",
              [r"$\alpha$", "Train reward", "Eval.\\ reward", "Eval.\\ std",
               "Discount rate", "``Never discount''", "Policy changes"],
              [[a, f3(summarise("", lr_results[a])["train_mean_reward"]),
                f3(summarise("", lr_results[a])["eval_mean_reward"]),
                f3(summarise("", lr_results[a])["eval_std_across_seeds"]),
                f3(summarise("", lr_results[a])["eval_discount_rate"]),
                f"{summarise('', lr_results[a])['seeds_with_never_discount_policy']}/10",
                f"{summarise('', lr_results[a])['mean_policy_flips']:.1f}"]
               for a in [0.01, 0.10, 0.50]], "rrrrrcr")
    tex_table(TAB / "exploration.tex",
              ["Exploration", "Train reward", "Train std", "Eval.\\ reward", "Eval.\\ std",
               "``Never discount''"],
              [[l.replace("ε", "$\\varepsilon$").replace("→", "$\\to$"),
                f3(summarise("", exp_results[l])["train_mean_reward"]),
                f3(summarise("", exp_results[l])["train_std_across_seeds"]),
                f3(summarise("", exp_results[l])["eval_mean_reward"]),
                f3(summarise("", exp_results[l])["eval_std_across_seeds"]),
                f"{summarise('', exp_results[l])['seeds_with_never_discount_policy']}/10"]
               for l, _ in EXPLORATION], "lrrrrc")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = ["RL α=0.10", "RL α=0.01", "RL α=0.50", "RL ε=0.05", "RL ε=0.30", "RL ε=0", "Rule"]
    vals = [[r["eval_mean_reward_per_step"] for r in rows] for _, rows in table]
    ax.boxplot(vals, tick_labels=names, widths=0.5)
    for i, v in enumerate(vals):
        ax.scatter(np.full(len(v), i + 1), v, s=10, color=COLORS[0], zorder=3)
    ax.set_ylabel("Evaluation mean reward per step")
    ax.set_title("Greedy evaluation against the rule-based opponent, 10 seeds")
    save(fig, "multi_seed_comparison.png")


def experiment_rl_vs_rl(config):
    print("\n== Question 8: two simultaneous learners ==")
    flips_rule, flips_rl, rows_out = [], [], []
    disc_curves_rule, disc_curves_rl = [], []
    example = None
    for seed in SEEDS:
        a_rule, _, rows_rule = train(config, seed, opponent="rule")
        a0, a1, rows_rl = train(config, seed, opponent="rl")
        if example is None:
            example = (a0, a1, rows_rl)
        flips_rule.append(policy_flips([r["rl_policy"] for r in rows_rule]))
        flips_rl.append(policy_flips([r["rl_policy"] for r in rows_rl]))
        disc_curves_rule.append(windowed([r["rl_discount_rate"] for r in rows_rule], config.window))
        disc_curves_rl.append(windowed([(r["rl_discount_rate"] + r["opponent_discount_rate"]) / 2
                                        for r in rows_rl], config.window))
        es = seed + EVAL_SEED_OFFSET
        rl_rl, _ = evaluate(config, a0, a1, es)
        rows_out.append({
            "seed": seed,
            "policy_flips_vs_rule": flips_rule[-1],
            "policy_flips_vs_learning_rl": flips_rl[-1],
            "rl_vs_rl_eval_reward_seat0": rl_rl[0]["mean_reward_per_step"],
            "rl_vs_rl_eval_reward_seat1": rl_rl[1]["mean_reward_per_step"],
            "rl_vs_rl_discount_rate_seat0": rl_rl[0]["discount_action_rate"],
            "rl_vs_rl_discount_rate_seat1": rl_rl[1]["discount_action_rate"],
            "late_flips_vs_rule": policy_flips([r["rl_policy"] for r in rows_rule[-2500:]]),
            "late_flips_vs_learning_rl": policy_flips([r["rl_policy"] for r in rows_rl[-2500:]]),
        })
    write_csv(OUT / "rl_vs_rl.csv", rows_out)
    with open(OUT / "q_tables.txt", "a", encoding="utf-8") as f:
        f.write(q_table_text(f"RL vs RL, seat 0 (seed {SEEDS[0]})", example[0]))
        f.write(q_table_text(f"RL vs RL, seat 1 (seed {SEEDS[0]})", example[1]))
    summary = {k: mean_std([r[k] for r in rows_out]) for k in rows_out[0] if k != "seed"}
    for k, v in summary.items():
        print(f"{k:35s} mean={v[0]:.3f} std={v[1]:.3f}")
    write_csv(OUT / "rl_vs_rl_summary.csv",
              [{"metric": k, "mean": v[0], "std": v[1]} for k, v in summary.items()])

    k = config.window
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(1, len(disc_curves_rule[0]) + 1) * k
    axes[0].plot(x, np.mean(disc_curves_rule, axis=0), color=COLORS[0], lw=2,
                 label="RL vs fixed rule (opponent stationary)")
    axes[0].plot(x, np.mean(disc_curves_rl, axis=0), color=COLORS[3], lw=2,
                 label="RL vs RL (both learning)")
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Discount rate of the learners")
    axes[0].set_title("Discount rate during training (mean over 10 seeds)")
    axes[0].legend(fontsize=8)

    rows_rl = example[2]
    pol0 = np.array([r["rl_policy"] for r in rows_rl])
    pol1 = np.array([r["opponent_policy"] for r in rows_rl])
    ep = np.arange(1, len(pol0) + 1)
    for i, (name, bit) in enumerate([("(0,0)", 1), ("(0,1)", 2), ("(1,0)", 4), ("(1,1)", 8)]):
        axes[1].step(ep, i * 2.5 + ((pol0 & bit) > 0), where="post", color=COLORS[0], lw=1)
        axes[1].step(ep, i * 2.5 + 1.1 + ((pol1 & bit) > 0) * 0.9, where="post",
                     color=COLORS[3], lw=1)
    axes[1].set_yticks([i * 2.5 + 0.5 for i in range(4)] + [i * 2.5 + 1.55 for i in range(4)],
                       [f"A0 {s}" for s in ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]] +
                       [f"A1 {s}" for s in ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]], fontsize=7)
    axes[1].set_xlabel("Training episode")
    axes[1].set_title(f"Greedy action per state, seed {SEEDS[0]} (high = discount)")
    axes[1].grid(False)
    save(fig, "rl_vs_rl.png")
    return summary


def experiment_modification(config):
    print("\n== Question 9: modified competitor (aggressive) ==")
    per_seed, curves, agents = run_config(config, "RL vs Aggressive", opponent="aggressive")
    rule_rows = rule_in_seat0(config, opponent="aggressive")

    # Baseline RL agent (trained against the reference rule) moved, unchanged,
    # into the new market: does what it learned transfer?
    transfer = []
    for seed in SEEDS:
        agent, _, _ = train(config, seed, opponent="rule")
        s, _ = evaluate(config, agent, AggressiveAgent(), seed + EVAL_SEED_OFFSET)
        transfer.append(s[0]["mean_reward_per_step"])

    with open(OUT / "q_tables.txt", "a", encoding="utf-8") as f:
        f.write(q_table_text(f"RL trained against the aggressive competitor (seed {SEEDS[0]})",
                             agents[0]))

    rl_s = summarise("RL (retrained)", per_seed)
    rule_s = summarise("Rule-based", rule_rows)
    tr_m, tr_s = mean_std(transfer)
    rows = [
        {"agent": "RL retrained against the aggressive competitor",
         "eval_mean": rl_s["eval_mean_reward"], "eval_std": rl_s["eval_std_across_seeds"],
         "discount_rate": rl_s["eval_discount_rate"]},
        {"agent": "RL trained on the reference market (not retrained)",
         "eval_mean": tr_m, "eval_std": tr_s, "discount_rate": float("nan")},
        {"agent": "Rule-based", "eval_mean": rule_s["eval_mean_reward"],
         "eval_std": rule_s["eval_std_across_seeds"],
         "discount_rate": rule_s["eval_discount_rate"]},
    ]
    write_csv(OUT / "modification_aggressive.csv", rows)
    write_csv(OUT / "modification_aggressive_per_seed.csv", per_seed + rule_rows)
    for r in rows:
        print(r)
    tex_table(TAB / "modification.tex",
              ["Agent in seat 0 (opponent: aggressive competitor)", "Mean reward", "Std.\\ dev.",
               "Discount rate"],
              [[r["agent"], f3(r["eval_mean"]), f3(r["eval_std"]),
                "--" if np.isnan(r["discount_rate"]) else f3(r["discount_rate"])] for r in rows],
              "lrrr")
    diffs = [a["eval_mean_reward_per_step"] - b["eval_mean_reward_per_step"]
             for a, b in zip(per_seed, rule_rows)]
    m, s = mean_std(diffs)
    print(f"Paired difference RL - Rule (aggressive market): mean={m:.3f}, std={s:.3f}")
    write_csv(OUT / "modification_paired_difference.csv",
              [{"seed": sd, "difference": d} for sd, d in zip(SEEDS, diffs)])
    return rows


OPPONENTS = ["rule", "aggressive", "tit_for_tat", "always_discount", "random"]


def experiment_robustness(config):
    print("\n== Question 10: robustness against a pool of opponents ==")
    candidates = {
        "Rule-based": lambda seed: RuleBasedAgent(),
        "RL baseline (trained vs rule)":
            lambda seed: train(config, seed, opponent="rule")[0],
        "RL + opponent pool + α=1/N":
            lambda seed: train(config, seed, step_size="visit", opponent_pool=OPPONENTS)[0],
        "Improved RL (+ own previous action in state)":
            lambda seed: train(config, seed, step_size="visit", opponent_pool=OPPONENTS,
                               memory=True)[0],
    }
    rows = []
    for name, build in candidates.items():
        for seed in SEEDS:
            agent = build(seed)
            for opp in OPPONENTS:
                s, _ = evaluate(config, agent, make_opponent(opp, seed + 1),
                                seed + EVAL_SEED_OFFSET + 1)
                rows.append({"agent": name, "seed": seed, "opponent": opp,
                             "eval_mean_reward_per_step": s[0]["mean_reward_per_step"],
                             "discount_rate": s[0]["discount_action_rate"]})
    write_csv(OUT / "robustness_per_seed.csv", rows)

    summary, tex_rows = [], []
    for name in candidates:
        cells, worst = [name.replace("α=1/N", "$\\alpha=1/N$")], []
        for opp in OPPONENTS:
            v = [r["eval_mean_reward_per_step"] for r in rows
                 if r["agent"] == name and r["opponent"] == opp]
            m, s = mean_std(v)
            summary.append({"agent": name, "opponent": opp, "mean": m, "std": s})
            cells.append(f"{m:.2f} $\\pm$ {s:.2f}")
            worst.append(m)
        cells.append(f"{np.mean(worst):.2f}")
        cells.append(f"{np.min(worst):.2f}")
        tex_rows.append(cells)
    # Reference row: best expected reward per step against each opponent,
    # computed from the model (never discount is optimal against the reactive
    # opponents; against stateless opponents the best action per demand level
    # follows from expected_profit with the opponent's discount probability).
    best = [35.0, 30.0, 35.0]
    for p_disc in (1.0, 0.5):  # always discount, random
        v = 0.0
        for d in range(2):
            v += 0.5 * max(p_disc * expected_profit(config, d, a, 1)
                           + (1 - p_disc) * expected_profit(config, d, a, 0) for a in range(2))
        best.append(v)
    tex_rows.append(["Best response (from the model)"] + [f"{b:.2f}" for b in best]
                    + [f"{np.mean(best):.2f}", f"{np.min(best):.2f}"])
    summary += [{"agent": "Best response (model)", "opponent": o, "mean": b, "std": 0.0}
                for o, b in zip(OPPONENTS, best)]
    write_csv(OUT / "robustness_summary.csv", summary)
    for r in summary:
        print(f"{r['agent']:38s} vs {r['opponent']:16s} {r['mean']:.3f} ± {r['std']:.3f}")
    tex_table(TAB / "robustness.tex",
              ["Agent", "Rule", "Aggressive", "Tit-for-tat", "Always disc.", "Random",
               "Average", "Worst"], tex_rows, "lrrrrrrr")


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():
    config = Config()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    for d in (OUT, FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AGENT02: RL AGENT VS RULE-BASED AGENT")
    print("=" * 60)
    print(config)
    print(f"Seeds for repeated runs: {SEEDS}")
    print(f"Output directory: {OUT}")

    # Optional: `python agent02_experiment.py robustness` reruns one part only.
    only = set(sys.argv[1:])
    if not only:
        (OUT / "q_tables.txt").unlink(missing_ok=True)
        experiment_model(config)
        experiment_a_b(config)
        lr = experiment_c(config)
        ex = experiment_d(config)
        experiment_e(config, lr, ex)
    if not only or "rl_vs_rl" in only:
        experiment_rl_vs_rl(config)
    if not only or "modification" in only:
        experiment_modification(config)
    if not only or "robustness" in only:
        experiment_robustness(config)

    print("\nALL EXPERIMENTS COMPLETED. Results saved to:", OUT)


if __name__ == "__main__":
    main()
