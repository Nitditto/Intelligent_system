# Agent 01: AI Agents and Reinforcement Learning

A report on AI agents and reinforcement learning, based on two sources, with one demo: a Q-learning agent that learns the shortest route through a 10 × 10 maze.

1. Google Cloud, "What are AI agents?" https://cloud.google.com/discover/what-are-ai-agents
2. GeeksforGeeks, "What is Reinforcement Learning?" https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/

| Path | Content |
|---|---|
| `notebook/agent_rl_demo.ipynb` | the demo, every step explained and run |
| `report/agent_01_truongnv_095.tex` / `.pdf` | the report (about 15 pages) |
| `PLAN.md` | the plan the report follows |

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name agent01 --display-name "Python (agent_01)"
jupyter nbconvert --to notebook --execute --inplace notebook/agent_rl_demo.ipynb   # also writes report/images/
cd report && xelatex agent_01_truongnv_095.tex && xelatex agent_01_truongnv_095.tex
```

Results (seed 42): the agent first reaches the goal in episode 721, succeeds 89% of the time over the last 500 episodes, and from the start follows the shortest route of 22 steps (reward 29). A random agent never reaches the goal.
