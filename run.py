"""
AlphaOptimizer: The outer training loop.

Runs multiple episodes, manages the population, and adaptively
controls exploration vs. exploitation across the entire search.

Usage:
    python run.py --alpha_id <seed_id>
    python run.py --alpha_id <seed_id> --episodes 5 --sim_budget 15
    python run.py --resume                              # load saved population
    python run.py --alpha_id <id> --mode explore        # force exploration mode
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────

# Local modules
sys.path.insert(0, str(Path(__file__).parent))

from ace_extensions import get_alpha_recordset, get_datafield, get_stored_session
from ace_lib import get_simulation_result_json
from alpha_agent import AlphaAgent, PromptBuilder
from alpha_env import AlphaAction, ActionType, build_hard_constraint
from alpha_environment import AlphaEnvironment
from alpha_utils import extract_datafields, fix_fastexpr, generate_pnl_chart
from population import AlphaCandidate, Population


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

POPULATION_FILE = "population.json"
RESULTS_FILE    = "results.jsonl"

with open("config.json") as f:
    CONFIG = json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: build datafield context string
# ─────────────────────────────────────────────────────────────────────────────

def build_datafield_context(brain_session, expression: str) -> str:
    datafields = extract_datafields(expression)
    groups = CONFIG.get("groups", [])

    parts = ["## Data Fields Available"]
    for name in datafields:
        if name in groups:
            continue
        info = get_datafield(brain_session, name)
        if "id" not in info:
            continue
        parts.append(f"- **{info['id']}** ({info['type']}): {info['description']}")

    parts.append("\n## Grouping Fields")
    parts.append(", ".join(g for g in groups if not g.startswith("!")))

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: parse agent output → AlphaAction
# ─────────────────────────────────────────────────────────────────────────────

def parse_agent_output(output: dict) -> AlphaAction:
    try:
        action_type = ActionType(output.get("action_type", "mutate_expression"))
    except ValueError:
        action_type = ActionType.MUTATE_EXPRESSION

    settings = {
        "decay":          max(int(output.get("decay", 4)), 1),   # enforce minimum decay=1
        "neutralization": str(output.get("neutralization", "SUBINDUSTRY")).strip(),
        "truncation":     float(output.get("truncation", 0.05)),
        "universe":       "TOP3000",   # hard locked — ignore whatever LLM says
    }

    try:
        confidence = float(output.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    return AlphaAction(
        action_type = action_type,
        expression  = fix_fastexpr(str(output.get("expression", ""))),
        settings    = settings,
        reasoning   = output.get("chain_of_thought", ""),
        confidence  = confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    env: AlphaEnvironment,
    agent: AlphaAgent,
    population: Population,
    seed_expression: str,
    seed_settings: dict,
    datafield_context: str,
    episode_id: str,
    sim_budget: int,
) -> dict:
    """Run a single episode. Returns summary dict."""

    console.rule(f"[bold cyan]Episode {episode_id[:8]}[/]")

    obs = env.reset(
        seed_expression = seed_expression,
        settings        = seed_settings,
        episode_id      = episode_id,
    )

    _log_obs(obs, step=0)

    while not obs.done:
        mode   = env.mode
        failing = obs.sim_result.failing_checks if obs.sim_result else []

        # Confidence-based mode override (suggestion #3):
        # if population is trending low-confidence, force explore
        if mode == "exploit" and population.should_force_explore(threshold=0.45):
            console.print("[dim]Low-confidence trend detected → overriding to EXPLORE[/]")
            mode = "explore"

        # Build system prompt targeted to current failing checks
        system_prompt = PromptBuilder.build_system_prompt(
            datafield_context = datafield_context,
            failing_checks    = failing,
            mode              = mode,
        )

        # Select parent alphas for cross-breed if in diversify mode
        parent_alphas = None
        if mode == "diversify":
            pair = population.select_diverse_pair()
            if pair:
                parent_alphas = [
                    {"expression": p.expression, "sharpe": p.sharpe}
                    for p in pair
                ]

        # Build hard constraint message (suggestion #1)
        hard_constraint = build_hard_constraint(failing, obs.sim_result)

        # Build user message with hard constraints injected at top
        user_msg = PromptBuilder.build_user_message(
            step               = obs.step,
            current_expression = env.current_expression,
            current_settings   = env.current_settings,
            sim_summary        = env.build_sim_summary(obs.sim_result),
            history_summary    = obs.history_summary,
            parent_alphas      = parent_alphas,
            hard_constraint    = hard_constraint,
        )

        # Generate action
        console.print(
            f"\n[yellow]▶ Mode: {mode.upper()} | Step {obs.step + 1}/{sim_budget} | "
            f"Best: {env.state.best_sharpe:.3f}[/]"
        )
        console.print("[dim]Agent thinking...[/]")
        agent_output = agent.generate(system_prompt, user_msg, mode=mode)

        action = parse_agent_output(agent_output)

        # Show reasoning
        cot = agent_output.get("chain_of_thought", "")
        if cot:
            console.print(Panel(
                cot[:600] + ("..." if len(cot) > 600 else ""),
                title="[dim]Chain of Thought[/]",
                border_style="dim",
            ))

        console.print(
            f"[cyan]Action:[/] {action.action_type.value} | "
            f"[cyan]Confidence:[/] {action.confidence:.2f}"
        )

        # Step the environment
        obs = env.step(action)
        _log_obs(obs, step=obs.step)

        # Add to population with confidence stored
        if obs.sim_result:
            candidate = AlphaCandidate(
                neg_reward  = -obs.reward,
                alpha_id    = obs.sim_result.alpha_id,
                expression  = env.current_expression,
                settings    = dict(env.current_settings),
                reward      = obs.reward,
                sharpe      = obs.sim_result.sharpe,
                fitness     = obs.sim_result.fitness,
                turnover    = obs.sim_result.turnover,
                passing     = obs.sim_result.passing,
                fingerprint = obs.sim_result.fingerprint,
                confidence  = action.confidence,
                episode_id  = episode_id,
            )
            added = population.try_add(candidate)
            if added:
                console.print(f"[green]★ Population: {population.statistics()['size']} elites[/]")

    # Episode summary
    pop_stats = population.statistics()
    summary = {
        "episode_id":       episode_id,
        "steps":            obs.step,
        "target_achieved":  env.state.target_achieved if env.state else False,
        "best_sharpe":      env.state.best_sharpe if env.state else 0.0,
        "best_alpha_id":    env.state.best_alpha_id if env.state else "",
        "population_stats": pop_stats,
        "final_message":    obs.system_message,
    }

    status = "[green]✓ SUCCESS[/]" if summary["target_achieved"] else "[yellow]∅ Budget exhausted[/]"
    console.print(Panel(
        f"{status}\n"
        f"Best Sharpe: {summary['best_sharpe']:.4f}\n"
        f"Best Alpha:  {summary['best_alpha_id']}\n"
        f"Population:  {pop_stats['size']} elites | {pop_stats['passing_count']} passing",
        title="[bold]Episode Complete[/]",
        border_style="cyan",
    ))

    return summary


def _log_obs(obs, step: int):
    """Pretty-print an observation."""
    if obs.sim_result:
        r = obs.sim_result
        fails = f"[red]{', '.join(r.failing_checks)}[/]" if r.failing_checks else "[green]ALL PASS[/]"
        console.print(
            f"  Step {step:02d} | "
            f"Sharpe={r.sharpe:.3f} Fitness={r.fitness:.3f} Turn={r.turnover:.3f} | "
            f"Reward={obs.reward:+.3f} | Checks: {fails}"
        )
    else:
        console.print(f"  Step {step:02d} | [red]No sim result[/] | Reward={obs.reward:+.3f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AlphaOptimizer: RL-style alpha generation")
    parser.add_argument("--alpha_id", "-id", type=str, help="Seed alpha ID from BRAIN")
    parser.add_argument("--episodes",        type=int, default=3,  help="Number of episodes")
    parser.add_argument("--sim_budget",      type=int, default=25, help="Max simulations per episode")
    parser.add_argument("--resume",          action="store_true",  help="Load saved population")
    parser.add_argument("--population_size", type=int, default=0, help="Elite archive size")
    parser.add_argument("--model",           type=str, default="gemini-2.5-flash")
    args = parser.parse_args()

    if not args.alpha_id and not args.resume:
        parser.error("Provide --alpha_id or --resume to continue from saved population.")

    # ── Setup ─────────────────────────────────────────────────────────────────
    gemini_keys = os.getenv("GEMINI_API_KEYS", "").split(",")
    gemini_keys = [k.strip() for k in gemini_keys if k.strip()]
    if not gemini_keys:
        console.print("[red]Error: GEMINI_API_KEYS not set in .env[/]")
        sys.exit(1)

    brain_session = get_stored_session(duration=10800)

    agent      = AlphaAgent(api_keys=gemini_keys, model=args.model)
    env        = AlphaEnvironment(brain_session=brain_session, sim_budget=args.sim_budget)
    population = Population(capacity=args.population_size)

    if args.resume and Path(POPULATION_FILE).exists():
        population.load(POPULATION_FILE)
        console.print(f"[green]Resumed population: {population.statistics()}[/]")

    # ── Seed alpha ────────────────────────────────────────────────────────────
    if args.alpha_id:
        console.print(f"[cyan]Loading seed alpha {args.alpha_id}...[/]")
        seed_sim = get_simulation_result_json(brain_session, args.alpha_id)
        seed_expression = seed_sim["regular"]["code"]
        seed_settings   = seed_sim["settings"]

        # Generate and save initial PnL chart
        pnl = get_alpha_recordset(brain_session, args.alpha_id, "pnl")
        generate_pnl_chart(CONFIG["pnl_chart"], pnl)

        datafield_context = build_datafield_context(brain_session, seed_expression)
    else:
        # Resume mode: use best from population as seed
        best = population.best(1)
        if not best:
            console.print("[red]Population empty. Provide --alpha_id.[/]")
            sys.exit(1)
        seed_expression   = best[0].expression
        seed_settings     = best[0].settings
        datafield_context = "## Data Fields\n(Resuming from saved population)"

    console.print(Panel(
        f"Seed Expression:\n{seed_expression}\n\n"
        f"Settings: {json.dumps(seed_settings, indent=2)}",
        title="[bold]Seed Alpha[/]",
        border_style="blue",
    ))

    # ── Episode loop ──────────────────────────────────────────────────────────
    all_results = []

    for ep in range(args.episodes):
        # Choose seed for this episode:
        # Episode 0: use original seed
        # Later episodes: sample from population (if non-empty) for diversity
        if ep > 0 and population.statistics()["size"] > 0:
            parent = population.tournament_select(k=5)
            ep_seed_expression = parent.expression
            ep_seed_settings   = parent.settings
            console.print(f"\n[dim]Episode {ep}: seeding from population (sharpe={parent.sharpe:.3f})[/]")
        else:
            ep_seed_expression = seed_expression
            ep_seed_settings   = seed_settings

        episode_id = str(uuid.uuid4())

        summary = run_episode(
            env               = env,
            agent             = agent,
            population        = population,
            seed_expression   = ep_seed_expression,
            seed_settings     = ep_seed_settings,
            datafield_context = datafield_context,
            episode_id        = episode_id,
            sim_budget        = args.sim_budget,
        )
        all_results.append(summary)

        # Save population after each episode
        population.save(POPULATION_FILE)

        # Append result to JSONL log
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(summary) + "\n")

        # Early stopping: if submissible alpha found, generate PnL chart
        if summary["target_achieved"] and summary["best_alpha_id"]:
            try:
                pnl = get_alpha_recordset(brain_session, summary["best_alpha_id"], "pnl")
                generate_pnl_chart(CONFIG["pnl_chart"], pnl)
                console.print(f"[green]PnL chart generated for {summary['best_alpha_id']}[/]")
            except Exception as e:
                console.print(f"[yellow]PnL chart failed: {e}[/]")

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_final_summary(all_results, population)


def _print_final_summary(results: list[dict], population: Population):
    console.rule("[bold]Final Summary[/]")

    table = Table(title="Episode Results", show_header=True)
    table.add_column("Episode", style="dim")
    table.add_column("Steps")
    table.add_column("Best Sharpe", style="cyan")
    table.add_column("Best Alpha ID", style="blue")
    table.add_column("Success", style="green")

    for r in results:
        table.add_row(
            r["episode_id"][:8],
            str(r["steps"]),
            f"{r['best_sharpe']:.4f}",
            r["best_alpha_id"][:12] if r["best_alpha_id"] else "—",
            "✓" if r["target_achieved"] else "✗",
        )
    console.print(table)

    pop_stats = population.statistics()
    console.print(Panel(
        f"Population size:     {pop_stats['size']}\n"
        f"Best sharpe ever:    {pop_stats['best_sharpe']:.4f}\n"
        f"Passing alphas:      {pop_stats['passing_count']}\n"
        f"Diversity (unique):  {pop_stats['diversity']}",
        title="[bold]Population Statistics[/]",
        border_style="green",
    ))

    # Print top-5 alphas
    top5 = population.best(5)
    if top5:
        console.print("\n[bold]Top 5 Alphas:[/]")
        for i, c in enumerate(top5, 1):
            status = "✓ PASSING" if c.passing else "✗"
            console.print(
                f"  {i}. {c.alpha_id[:12]} | "
                f"Sharpe={c.sharpe:.3f} | Fitness={c.fitness:.3f} | "
                f"Reward={c.reward:.3f} | {status}"
            )
            console.print(f"     [dim]{c.expression[:120]}...[/]" if len(c.expression) > 120 else f"     [dim]{c.expression}[/]")


if __name__ == "__main__":
    main()