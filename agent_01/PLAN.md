# Agent 01 — Plan

**Topic:** AI agents and Reinforcement Learning, with one demo.
**Sources (only these two):**
[1] Google Cloud. "What are AI agents?" https://cloud.google.com/discover/what-are-ai-agents
[2] GeeksforGeeks. "What is Reinforcement Learning?" https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/

**Deliverables**

```
agent_01/
├── PLAN.md
├── requirements.txt              numpy, matplotlib
├── notebook/agent_rl_demo.ipynb  runs the demo, every step explained
└── report/agent_01_truongnv_095.tex / .pdf   (+ images/ from the notebook)
```

## 1. The demo: a Q-learning agent in a maze

Taken from the worked example in [2], and read as an AI agent in the sense of [1]:

| Agent idea from [1] | The same thing in the maze demo [2] |
|---|---|
| Observing | the agent sees its current cell (the **state**) |
| Reasoning / Planning | it picks a move with its **policy** (ε-greedy over the Q-table) |
| Acting | it moves left / right / up / down (the **action**) |
| Self-refining (learning from feedback) | the **reward** updates the Q-table |
| Memory | the Q-table is what the agent remembers between episodes |

Settings from [2]:
- a 10×10 maze (0 = path, 1 = obstacle), start (0,0), goal (9,9)
- 5,000 episodes, α = 0.1, γ = 0.9, ε = 0.5, decaying by ×0.995 to a minimum of 0.01
- rewards: −1 per step, −10 for hitting a wall or obstacle, +50 at the goal

## 2. Notebook (`agent_rl_demo.ipynb`)

Each part has explanation cells (from scratch, explaining why, no long formulas) followed by code.

1. **What an agent is:** the observe → decide → act → learn loop [1], and how RL fits into it [2].
2. **The environment:** build the maze and draw it.
3. **Before learning:** an agent with a random policy wanders and almost never reaches the goal. This is the baseline.
4. **Q-learning:** the Q-table, the ε-greedy choice (exploration vs exploitation), the update rule explained in words with one worked step on real numbers, then training.
5. **Results:**
   - reward per episode (the learning curve)
   - steps per episode
   - ε over time
   - the learned path drawn on the maze
   - a map of the best action in each cell
6. **Comparison:** random vs trained agent (success rate and number of steps over 100 runs).
7. **Limitations** [2]: reward design, exploration vs exploitation, the cost of training. Link these to the agent challenges in [1].

## 3. Report (`agent_01_truongnv_095.pdf`, about 12–15 pages, English, same cover as before)

1. **Introduction:** the goal and the two sources.
2. **AI agents** [1]:
   - definition
   - key features (reasoning, acting, observing, planning, collaborating, self-refining)
   - agents vs assistants vs bots (comparison table)
   - how agents work (persona, memory, tools, model)
   - types (interactive / background; single / multi-agent)
   - benefits and challenges
   - use cases (customer, employee, creative, data, code, security)
3. **Reinforcement Learning** [2]:
   - definition and core terms (agent, environment, state, action, reward)
   - components (policy, reward signal, value function, model)
   - how RL works: the loop, exploration vs exploitation, MDP
   - positive and negative reinforcement
   - online vs offline RL
   - applications, advantages and limitations
4. **Connecting the two:** an RL agent is an AI agent that learns from rewards (the table in §1 of this plan).
5. **Demo:** the maze setup, Q-learning, the results with figures from the notebook (learning curve, learned path, random vs trained), and an explanation of what the agent learned and why.
6. **Discussion and conclusion.**
7. **References:** [1], [2].

## 4. Open questions

1. Folder `agent_01/`, report file name `agent_01_truongnv_095.pdf` (student name and ID: Nguyễn Văn Trường, B23DCCE095). Is that right?
2. Report and notebook in English, like the previous assignments?
3. Is one demo (the maze) enough, or do you also want something simple for the AI-agent side, e.g. a rule-based bot compared with the learning agent?
