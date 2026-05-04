"""Post-processing plots for colocation mining results."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colocation import ColocationResult, ColocationRule


def _format_colocation(features: tuple[str, ...]) -> str:
    return " | ".join(features)


def _format_rule(rule: ColocationRule) -> str:
    return (
        f"{_format_colocation(rule.antecedent)} => "
        f"{_format_colocation(rule.consequent)}"
    )


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _empty_panel(ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="0.4",
    )
    ax.set_axis_off()


SPATIAL_STYLE_PAIRS: tuple[tuple[str, str], ...] = (
    ("#D7191C", "o"),  # red circle
    ("#2C7BB6", "s"),  # blue square
    ("#1A9641", "^"),  # green triangle-up
    ("#F57C00", "D"),  # orange diamond
    ("#7B3294", "P"),  # purple plus-filled
    ("#008080", "X"),  # teal x-filled
    ("#4D4D4D", "v"),  # dark gray triangle-down
    ("#A65628", "<"),  # brown triangle-left
    ("#E7298A", ">"),  # magenta triangle-right
    ("#66A61E", "*"),  # olive star
)


def _choose_colocations_for_spatial_panels(
    prevalent: dict[tuple[str, ...], float],
    max_colocations: int | None,
) -> list[tuple[tuple[str, ...], float]]:
    """Pick colocations with priority on covering as many sizes as possible."""
    ranked = sorted(
        prevalent.items(),
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )
    if not ranked:
        return []

    best_per_size: dict[int, tuple[tuple[str, ...], float]] = {}
    for colocation, pi in ranked:
        size = len(colocation)
        if size not in best_per_size:
            best_per_size[size] = (colocation, pi)

    selected: list[tuple[tuple[str, ...], float]] = [
        best_per_size[size] for size in sorted(best_per_size)
    ]
    if max_colocations is None:
        return selected

    if len(selected) >= max_colocations:
        return selected[:max_colocations]

    seen = {item[0] for item in selected}
    for item in ranked:
        if item[0] in seen:
            continue
        selected.append(item)
        seen.add(item[0])
        if len(selected) >= max_colocations:
            break
    return selected


def save_result_summary_plot(
    result: ColocationResult,
    output: Path,
    dataset_name: str,
    top_colocations: int = 15,
    top_rules: int = 15,
) -> Path:
    """Save a 2x2 summary figure for mined colocations and rules."""
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    ax_sizes, ax_top_colocs, ax_scatter, ax_top_rules = axs.flat

    # 1) How many prevalent patterns were found at each size.
    size_counts = Counter(len(c) for c in result.prevalent)
    if size_counts:
        sizes = sorted(size_counts)
        counts = [size_counts[s] for s in sizes]
        ax_sizes.bar(sizes, counts, color="#4C78A8")
        ax_sizes.set_xticks(sizes)
        ax_sizes.set_xlabel("Colocation size")
        ax_sizes.set_ylabel("Count")
        ax_sizes.set_title("Prevalent colocations by size")
        ax_sizes.grid(True, axis="y", linestyle=":", alpha=0.4)
        for x, y in zip(sizes, counts):
            ax_sizes.text(x, y, str(y), ha="center", va="bottom", fontsize=8)
    else:
        _empty_panel(
            ax_sizes,
            "Prevalent colocations by size",
            "No prevalent colocations found.",
        )

    # 2) Strongest colocations by participation index.
    top_colocs = sorted(
        result.prevalent.items(),
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )[:top_colocations]
    if top_colocs:
        labels = [_truncate(_format_colocation(c), 58) for c, _ in top_colocs]
        values = [pi for _, pi in top_colocs]
        y_pos = list(range(len(labels)))
        ax_top_colocs.barh(y_pos, values, color="#2A9D8F")
        ax_top_colocs.set_yticks(y_pos, labels=labels)
        ax_top_colocs.invert_yaxis()
        ax_top_colocs.set_xlim(0.0, 1.0)
        ax_top_colocs.set_xlabel("Participation index (PI)")
        ax_top_colocs.set_title(f"Top {len(top_colocs)} colocations by PI")
        ax_top_colocs.grid(True, axis="x", linestyle=":", alpha=0.4)
    else:
        _empty_panel(
            ax_top_colocs,
            "Top colocations by PI",
            "No prevalent colocations available.",
        )

    # 3) Rule quality cloud: cp vs prevalence (PI), grouped by integer rule size.
    if result.rules:
        rule_sizes = sorted(
            {len(r.antecedent) + len(r.consequent) for r in result.rules}
        )
        cmap = plt.get_cmap("tab20", max(len(rule_sizes), 1))
        for idx, rule_size in enumerate(rule_sizes):
            subset = [
                r
                for r in result.rules
                if len(r.antecedent) + len(r.consequent) == rule_size
            ]
            cp_vals = [r.conditional_probability for r in subset]
            pi_vals = [r.prevalence for r in subset]
            ax_scatter.scatter(
                cp_vals,
                pi_vals,
                color=cmap(idx),
                alpha=0.8,
                s=30,
                edgecolors="none",
                label=f"size {rule_size} ({len(subset)})",
            )
        ax_scatter.set_xlim(0.0, 1.02)
        ax_scatter.set_ylim(0.0, 1.02)
        ax_scatter.set_xlabel("Conditional probability (cp)")
        ax_scatter.set_ylabel("Prevalence (PI)")
        ax_scatter.set_title("Rule quality distribution")
        ax_scatter.grid(True, linestyle=":", alpha=0.4)
        ax_scatter.legend(title="Rule size", frameon=False, fontsize="small")
    else:
        _empty_panel(ax_scatter, "Rule quality distribution", "No rules generated.")

    # 4) Strongest rules by conditional probability.
    strongest_rules = sorted(
        result.rules,
        key=lambda r: (-r.conditional_probability, -r.prevalence),
    )[:top_rules]
    if strongest_rules:
        labels = [_truncate(_format_rule(rule), 70) for rule in strongest_rules]
        values = [rule.conditional_probability for rule in strongest_rules]
        y_pos = list(range(len(labels)))
        ax_top_rules.barh(y_pos, values, color="#F4A261")
        ax_top_rules.set_yticks(y_pos, labels=labels)
        ax_top_rules.invert_yaxis()
        ax_top_rules.set_xlim(0.0, 1.0)
        ax_top_rules.set_xlabel("Conditional probability (cp)")
        ax_top_rules.set_title(f"Top {len(strongest_rules)} rules by cp")
        ax_top_rules.grid(True, axis="x", linestyle=":", alpha=0.4)
    else:
        _empty_panel(ax_top_rules, "Top rules by cp", "No rules generated.")

    fig.suptitle(f"Colocation mining summary: {dataset_name}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def save_spatial_colocations_plot(
    result: ColocationResult,
    events: pd.DataFrame,
    output: Path,
    dataset_name: str,
    max_colocations: int | None = None,
) -> Path:
    """Save a spatial overlay plot for prevalent colocations across sizes."""
    ranked = _choose_colocations_for_spatial_panels(
        result.prevalent,
        max_colocations=max_colocations,
    )

    if not ranked:
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))
        ax.scatter(
            events["x"],
            events["y"],
            s=7,
            color="0.75",
            alpha=0.65,
            edgecolors="none",
        )
        ax.set_title(f"{dataset_name}: no prevalent colocations to highlight")
        ax.set_xlabel("x (meters, projected CRS)")
        ax.set_ylabel("y (meters, projected CRS)")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return output

    n_panels = len(ranked)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows), squeeze=False)
    axes = list(axs.flat)

    for panel_idx, ((colocation, pi), ax) in enumerate(zip(ranked, axes)):
        ax.scatter(events["x"], events["y"], s=6, color="0.88", alpha=0.5, edgecolors="none")
        table = result.table_instances.get(colocation, [])
        arr = np.asarray(table, dtype=np.int64)
        for feat_idx, feature in enumerate(colocation):
            if arr.size == 0:
                continue
            unique_indices = np.unique(arr[:, feat_idx])
            points = events.iloc[unique_indices]
            color, marker = SPATIAL_STYLE_PAIRS[feat_idx % len(SPATIAL_STYLE_PAIRS)]
            ax.scatter(
                points["x"],
                points["y"],
                s=42,
                alpha=0.95,
                color=color,
                marker=marker,
                linewidths=0.9,
                edgecolors="black",
                label=f"{feature} ({len(points)})",
            )

        ax.set_title(
            f"{panel_idx + 1}. {_truncate(_format_colocation(colocation), 75)}\n"
            f"PI={pi:.3f} | table rows={len(table)}"
        )
        ax.set_xlabel("x (meters, projected CRS)")
        ax.set_ylabel("y (meters, projected CRS)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="best", fontsize="small", frameon=False)

    for ax in axes[n_panels:]:
        ax.set_axis_off()

    fig.suptitle(
        f"Spatial point overlays across colocation sizes: {dataset_name}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output
