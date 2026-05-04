"""Post-processing plots for colocation mining results."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
