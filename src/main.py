"""CLI front-end for the colocation miner.

Loads an events CSV produced by ``dataset.py`` and runs the
Huang/Shekhar/Xiong colocation algorithm. Prints the prevalent
colocations (grouped by size) and the strongest rules, and optionally
writes per-table CSVs and a summary plot.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import typer

from colocation import ColocationResult, discover_colocations

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RESULTS_DIR = PROJ_ROOT / "results"

app = typer.Typer(
    help="Discover spatial colocation patterns (Huang/Shekhar/Xiong, 2004)."
)


def _format_features(features: tuple[str, ...]) -> str:
    return "{" + ", ".join(features) + "}"


def _export_name_prefix(csv_path: Path) -> str:
    stem = csv_path.stem
    for suffix in ("_osm_events", "_events"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _print_summary(result: ColocationResult, top_rules: int) -> None:
    typer.echo("")
    typer.echo("=" * 72)
    typer.echo(
        f"Mined {len(result.prevalent)} prevalent colocations "
        f"(size>=2, max size={result.parameters['max_size']})"
    )
    typer.echo("=" * 72)

    by_size: dict[int, list[tuple[tuple[str, ...], float]]] = {}
    for c, pi in result.prevalent.items():
        by_size.setdefault(len(c), []).append((c, pi))
    for size in sorted(by_size):
        typer.echo(f"\nSize {size}  ({len(by_size[size])} prevalent):")
        rows = sorted(by_size[size], key=lambda x: (-x[1], x[0]))
        for c, pi in rows:
            pr = result.participation_ratios[c]
            pr_str = "  ".join(f"pr({f})={pr[f]:.3f}" for f in c)
            typer.echo(f"  PI={pi:.3f}  {_format_features(c)}    {pr_str}")

    typer.echo("")
    typer.echo("=" * 72)
    typer.echo(
        f"Rules with cp >= {result.parameters['min_conditional_prob']:g}: "
        f"{len(result.rules)}"
    )
    typer.echo("=" * 72)
    rules_sorted = sorted(
        result.rules,
        key=lambda r: (-r.conditional_probability, -r.prevalence),
    )
    for rule in rules_sorted[:top_rules]:
        typer.echo(
            f"  {_format_features(rule.antecedent)} => "
            f"{_format_features(rule.consequent)}    "
            f"PI={rule.prevalence:.3f}  cp={rule.conditional_probability:.3f}"
        )
    if len(rules_sorted) > top_rules:
        typer.echo(f"  ... ({len(rules_sorted) - top_rules} more rules omitted)")


def _write_csvs(result: ColocationResult, out_dir: Path, input_stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for c, pi in result.prevalent.items():
        pr = result.participation_ratios[c]
        rows.append(
            {
                "colocation": " | ".join(c),
                "size": len(c),
                "participation_index": pi,
                "min_pr_feature": min(pr, key=lambda f: pr[f]),
                "table_instances": len(result.table_instances[c]),
                **{f"pr[{f}]": v for f, v in pr.items()},
            }
        )
    pd.DataFrame(rows).sort_values(
        ["size", "participation_index"], ascending=[True, False]
    ).to_csv(out_dir / f"{input_stem}_prevalent_colocations.csv", index=False)

    pd.DataFrame(
        [
            {
                "antecedent": " | ".join(r.antecedent),
                "consequent": " | ".join(r.consequent),
                "rule_size": len(r.antecedent) + len(r.consequent),
                "prevalence": r.prevalence,
                "conditional_probability": r.conditional_probability,
            }
            for r in result.rules
        ]
    ).sort_values(
        ["rule_size", "conditional_probability"], ascending=[True, False]
    ).to_csv(
        out_dir / f"{input_stem}_rules.csv", index=False
    )

    typer.echo(f"\nWrote results to {out_dir}")


@app.command()
def run(
    csv_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Events CSV (output of dataset.py).",
    ),
    distance: float = typer.Option(
        100.0,
        "--distance",
        "-d",
        min=0.0,
        help="Neighbor distance threshold in CRS units (typically meters).",
    ),
    prevalence: float = typer.Option(
        0.3,
        "--prevalence",
        "-p",
        min=0.0,
        max=1.0,
        help="Minimum participation index (PI).",
    ),
    conditional: float = typer.Option(
        0.5,
        "--conditional",
        "-c",
        min=0.0,
        max=1.0,
        help="Minimum conditional probability for emitted rules.",
    ),
    min_count: int = typer.Option(
        0,
        "--min-count",
        min=0,
        help="Drop feature types with fewer than this many instances.",
    ),
    drop: list[str] = typer.Option(
        [],
        "--drop",
        help="Feature_type values to exclude (repeatable).",
    ),
    top_rules: int = typer.Option(
        30, "--top-rules", help="Number of strongest rules to print."
    ),
    output_dir: Path = typer.Option(
        RESULTS_DIR,
        "--output-dir",
        "-o",
        help="Directory for CSV outputs.",
    ),
    save_csv: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Write CSVs of prevalent colocations and rules.",
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        help="Save a PNG summary of mined colocations and rules.",
    ),
) -> None:
    """Run the colocation miner on an events CSV."""
    events = pd.read_csv(csv_path)
    typer.echo(f"Loaded {len(events):,} events from {csv_path}")
    counts = events["feature_type"].value_counts()
    typer.echo(f"Distinct feature types: {len(counts)}")

    drop_set = set(drop)
    if min_count > 0:
        drop_set |= set(counts[counts < min_count].index)
    if drop_set:
        typer.echo(f"Dropping {len(drop_set)} feature types: {sorted(drop_set)}")
        feature_filter = lambda f: f not in drop_set  # noqa: E731
    else:
        feature_filter = None

    started = time.perf_counter()
    result = discover_colocations(
        events,
        distance=distance,
        min_prevalence=prevalence,
        min_conditional_prob=conditional,
        feature_filter=feature_filter,
        progress_callback=lambda m: typer.echo(f"  {m}"),
    )
    elapsed = time.perf_counter() - started
    typer.echo(f"\nDone in {elapsed:.2f}s.")

    _print_summary(result, top_rules=top_rules)

    input_stem = _export_name_prefix(csv_path)
    if save_csv:
        _write_csvs(result, output_dir, input_stem)

    if plot:
        from plot import save_result_summary_plot

        top_n = min(max(top_rules, 10), 40)
        plot_path = output_dir / f"{input_stem}_summary.png"
        save_result_summary_plot(
            result=result,
            output=plot_path,
            dataset_name=input_stem,
            top_colocations=top_n,
            top_rules=top_n,
        )
        typer.echo(f"Saved summary plot to {plot_path}")


if __name__ == "__main__":
    app()
