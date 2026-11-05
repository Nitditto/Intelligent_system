# Agent 02: RL-Based Agents versus Rule-Based Agents

Laboratory Experiment 02 (due 29.10.2026). Two sellers in a simulated e-commerce market decide each step whether to give a 20% discount. A tabular Q-learning agent is compared with a fixed rule-based agent. The assignment, reference code and slides are in `material/`.

| Path | Content |
|---|---|
| `code/agent02_experiment.py` | environment, both agents, training, separate evaluation, every experiment (A–E, Q8–Q10) |
| `notebook/agent02_demo.ipynb` | the same experiments run step by step, with explanations |
| `results/` | CSV results, `q_tables.txt`, `figures/*.png`, `tables/*.tex` (the report `\input`s these tables) |
| `report/agent02_E23CNPM02_truongnv_095.tex` / `.pdf` | the report: setup, results, answers to the 10 questions, discussion |

```bash
python -m pip install -r requirements.txt
cd code && python agent02_experiment.py          # about 25 min, rewrites ../results/
cd ../report && xelatex agent02_E23CNPM02_truongnv_095.tex && xelatex agent02_E23CNPM02_truongnv_095.tex
```

The script keeps the reference environment, Q-learning update and rule. It adds:

- a fair evaluation: both agent types sit in seat 0 against the same opponent, with the same seed and the same demand sequence;
- 10 seeds per configuration;
- fixed-ε and greedy exploration settings;
- experiments for Q8 (RL vs RL), Q9 (aggressive competitor) and Q10 (opponent pool, α = 1/N, robustness test).

Main results (greedy evaluation, mean ± std over 10 seeds, reward per step):

| Agent | Reference market | Aggressive rival | Average over 5 rival types |
|---|---|---|---|
| Rule-based | 34.97 ± 0.25 | 27.84 | 29.94 |
| RL, reference settings | 34.39 ± 1.24 | 28.76 (retrained) | 31.08 |
| RL, opponent pool + α = 1/N | 34.72 ± 0.77 | 29.83 | 31.59 |
| Best response (model) | 35.00 | 30.00 | 31.80 |
