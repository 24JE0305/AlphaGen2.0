# ⚛️ AlphaOptimizer: RL-Style WorldQuant BRAIN Alpha Generation

## What This Is

A complete redesign of the naive `refine.py` linear loop into a proper **RL-style search system**.

### The Core Insight

`refine.py` treats alpha refinement as a **chain** (one alpha, iterate forward).  
This system treats it as a **tree search** (population of candidates, branching exploration).

```
refine.py:       seed → v1 → v2 → v3 → done
AlphaOptimizer:  seed → [v1.1, v1.2, v1.3]
                          ↓       ↓      ↓
                        [v2.x] [v2.y] [v2.z]  ← population grows
                          ↓
                       elite archive (top-50 kept)
```

---

## Architecture

```
run.py                     ← Main entrypoint + episode loop
├── alpha_environment.py   ← RL environment (state, step, reward)
├── alpha_env.py           ← Data types (SimResult, AlphaState, etc.)
├── alpha_agent.py         ← LLM policy (dynamic prompts, key rotation)
├── population.py          ← Elite archive (min-heap + diversity filter)
└── system_instructions/
    └── warnings.txt       ← Advanced quantitative playbook (NOT the bad one)
```

---

## Key Upgrades Over `refine.py`

| Feature                  | `refine.py`          | AlphaOptimizer              |
|--------------------------|----------------------|-----------------------------|
| Search strategy          | Linear chain         | Population + episode tree   |
| Reward signal            | None (just LLM)      | Shaped multi-objective      |
| Mode switching           | Static               | Adaptive (explore/exploit)  |
| System prompt            | Fixed                | Dynamic (targets failures)  |
| Operator injection       | All at once          | Lazy (by failing check)     |
| Diversity               | None                 | Fingerprint deduplication   |
| Cross-breeding           | No                   | Yes (LLM-guided mix)        |
| Resumability             | No                   | Yes (population.json)       |
| Key rotation             | Yes                  | Yes + exponential backoff   |

---

## Reward Function

```
reward = sharpe_reward     × 0.45   # log-scaled (harder near the top)
       + check_improvement × 0.25   # delta in failing checks
       + fitness_bonus     × 0.15   # rewards high fitness scores
       + novelty_bonus     × 0.05   # reward for visiting new regions
       - plateau_penalty   × 0.05   # discourages stagnation
       + completion_bonus  = 2.0    # large terminal reward on success
```

---

## Setup

```bash
# Same dependencies as before
pip install -r requirements.txt

# .env file
BRAIN_CREDENTIAL_EMAIL=your_email
BRAIN_CREDENTIAL_PASSWORD=your_password
GEMINI_API_KEYS=key1,key2,key3   # multiple keys = automatic rotation
```

---

## Usage

```bash
# Basic: 3 episodes × 15 simulations each
python run.py --alpha_id YOUR_ALPHA_ID

# More thorough: 5 episodes × 20 simulations each
python run.py --alpha_id YOUR_ALPHA_ID --episodes 5 --sim_budget 20

# Resume from saved population (no seed needed)
python run.py --resume

# Use a different model
python run.py --alpha_id YOUR_ALPHA_ID --model gemini-2.5-pro
```

---

## Adaptive Mode Switching

The agent operates in three modes, switching automatically:

| Mode         | When                              | Temperature | Strategy              |
|-------------|-----------------------------------|-------------|------------------------|
| `explore`   | Sharpe < 0.5, plateau > 3 steps  | 0.9         | Full rewrites welcome  |
| `exploit`   | Sharpe > 1.0, checks improving   | 0.2         | Minimal targeted fixes |
| `diversify` | Every 6 steps (forced)           | 0.6         | Cross-breed 2 parents  |

---

## Output Files

- `population.json` — Elite archive (top-50 alphas, resumable)
- `results.jsonl`   — Episode-by-episode results log
- `pnl_chart.png`   — PnL chart of latest best alpha

---

## Connection to RL Concepts

| RL Concept          | Implementation Here                                   |
|--------------------|-------------------------------------------------------|
| State               | `AlphaObservation` (expression + sim results)         |
| Action              | `AlphaAction` (mutate/rewrite/settings/cross-breed)  |
| Reward              | `RewardCalculator.compute()` (shaped, multi-objective)|
| Policy              | `AlphaAgent` (Gemini LLM with dynamic prompts)       |
| Episode             | One alpha → N simulations → done                     |
| Population          | `Population` (elite archive, tournament selection)   |
| Exploration         | Mode switching + temperature control                  |