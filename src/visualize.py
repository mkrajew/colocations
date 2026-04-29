from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import typer

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"

app = typer.Typer(help="Plot the OSM events CSV produced by dataset.py.")


def load_events(csv_path: Path) -> pd.DataFrame:
    """Read the events CSV and validate that the expected columns are present.

    Args:
        csv_path: Path to a CSV with columns ``instance_id, feature_type, x, y``
            (the format produced by ``dataset.py``).

    Returns:
        The loaded ``DataFrame``.

    Raises:
        typer.Exit: If the file is missing or does not contain the required
            columns.
    """
    if not csv_path.exists():
        typer.echo(f"File not found: {csv_path}", err=True)
        raise typer.Exit(code=1)

    df = pd.read_csv(csv_path)
    required = {"instance_id", "feature_type", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        typer.echo(f"CSV missing required columns: {sorted(missing)}", err=True)
        raise typer.Exit(code=1)
    return df


def filter_events(
    df: pd.DataFrame,
    feature_types: list[str] | None,
    keys: list[str] | None,
    bbox: tuple[float, float, float, float] | None,
    sample: int | None,
    seed: int,
) -> pd.DataFrame:
    """Apply a chain of optional filters to the events DataFrame.

    Filters are applied in the following order:

    1. Exact ``feature_type`` match (e.g. ``amenity=cafe``).
    2. Top-level key match (e.g. ``amenity``, ``shop``).
    3. Spatial bounding box ``(xmin, ymin, xmax, ymax)`` in the same metric
       CRS that ``dataset.py`` projected to.
    4. Random subsample of ``sample`` rows (taken last so it works on the
       already-filtered subset).

    Args:
        df: The full events frame.
        feature_types: Allowed ``"key=value"`` labels, or ``None`` for all.
        keys: Allowed top-level keys, or ``None`` for all.
        bbox: Optional spatial window.
        sample: Optional cap on number of rows to keep (random sample).
        seed: RNG seed for reproducible sampling.

    Returns:
        A new filtered ``DataFrame`` (the original is not modified).
    """
    out = df

    if feature_types:
        out = out[out["feature_type"].isin(feature_types)]

    if keys:
        top_level = out["feature_type"].str.split("=").str[0]
        out = out[top_level.isin(keys)]

    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        out = out[
            (out["x"] >= xmin)
            & (out["x"] <= xmax)
            & (out["y"] >= ymin)
            & (out["y"] <= ymax)
        ]

    if sample is not None and len(out) > sample:
        out = out.sample(n=sample, random_state=seed)

    return out.reset_index(drop=True)


MARKERS = (
    "o",
    "s",
    "^",
    "v",
    "D",
    "P",
    "X",
    "*",
    "<",
    ">",
    "p",
    "h",
    "8",
    "d",
)


def plot_events(
    df: pd.DataFrame,
    color_by: str,
    title: str,
    point_size: float,
    alpha: float,
    output: Path | None,
) -> None:
    """Render a scatter plot of the events, colored by category.

    Each category gets both a distinct color (from the ``tab20`` palette,
    cycling when there are more than twenty categories) and a distinct
    marker shape (fourteen shapes cycle when needed).

    Args:
        df: The (already filtered) events frame.
        color_by: Either ``"feature_type"`` for fine-grained labels, or
            ``"key"`` to color by top-level key only.
        title: Title shown above the plot.
        point_size: Marker area passed to ``scatter``.
        alpha: Marker transparency in [0, 1].
        output: If given, save the figure to this path instead of opening
            an interactive window.
    """
    if color_by == "key":
        df = df.assign(_color_col=df["feature_type"].str.split("=").str[0])
    else:
        df = df.assign(_color_col=df["feature_type"])

    fig, ax = plt.subplots(figsize=(10, 10))

    categories = sorted(df["_color_col"].unique())
    cmap = plt.get_cmap("tab20", max(len(categories), 1))

    for i, cat in enumerate(categories):
        sub = df[df["_color_col"] == cat]
        color = cmap(i % cmap.N)
        marker = MARKERS[i % len(MARKERS)]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=point_size,
            alpha=alpha,
            color=color,
            marker=marker,
            label=f"{cat} ({len(sub)})",
            edgecolor="white",
            linewidths=0.35,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (meters, projected CRS)")
    ax.set_ylabel("y (meters, projected CRS)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)

    if categories:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize="small",
            frameon=False,
        )

    fig.tight_layout()

    if output is None:
        plt.show()
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        typer.echo(f"Saved plot to {output}")
        plt.close(fig)


def parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    """Parse a ``"xmin,ymin,xmax,ymax"`` string into a 4-tuple of floats."""
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter(
            "Expected four comma-separated numbers: xmin,ymin,xmax,ymax."
        )
    try:
        xmin, ymin, xmax, ymax = (float(p) for p in parts)
    except ValueError as exc:
        raise typer.BadParameter(f"Could not parse bbox numbers: {exc}") from exc
    if xmin >= xmax or ymin >= ymax:
        raise typer.BadParameter("Bounding box must satisfy xmin<xmax and ymin<ymax.")
    return xmin, ymin, xmax, ymax


@app.command()
def show(
    csv_path: Path = typer.Argument(
        ...,
        help="Path to the events CSV produced by dataset.py.",
        exists=False,
    ),
    feature_type: list[str] = typer.Option(
        None,
        "--feature-type",
        "-f",
        help="Restrict to specific 'key=value' labels (repeatable).",
    ),
    key: list[str] = typer.Option(
        None,
        "--key",
        "-k",
        help="Restrict to specific top-level keys, e.g. amenity, shop (repeatable).",
    ),
    bbox: str = typer.Option(
        None,
        "--bbox",
        help="Spatial window 'xmin,ymin,xmax,ymax' in the CSV's projected CRS.",
    ),
    sample: int = typer.Option(
        None,
        "--sample",
        "-n",
        help="Randomly sample at most N rows after filtering.",
        min=1,
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for the random sample."),
    color_by: str = typer.Option(
        "key",
        "--color-by",
        "-c",
        help="Color points by 'key' (top-level) or 'feature_type' (fine-grained).",
    ),
    point_size: float = typer.Option(6.0, "--point-size", "-s", help="Marker size."),
    alpha: float = typer.Option(0.6, "--alpha", "-a", min=0.0, max=1.0),
    title: str = typer.Option(None, "--title", help="Plot title."),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="If given, save the plot to this file instead of opening a window.",
    ),
) -> None:
    """Plot a (subset of the) events CSV as a 2D scatter plot."""
    if color_by not in {"key", "feature_type"}:
        raise typer.BadParameter("--color-by must be 'key' or 'feature_type'.")

    df = load_events(csv_path)
    typer.echo(f"Loaded {len(df)} rows from {csv_path}.")

    bbox_tuple = parse_bbox(bbox)
    filtered = filter_events(
        df,
        feature_types=feature_type or None,
        keys=key or None,
        bbox=bbox_tuple,
        sample=sample,
        seed=seed,
    )

    if len(filtered) == 0:
        typer.echo("No rows left after filtering — nothing to plot.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Plotting {len(filtered)} of {len(df)} rows.")

    if title is None:
        title = f"{csv_path.stem} ({len(filtered)} of {len(df)} events)"

    plot_events(
        filtered,
        color_by=color_by,
        title=title,
        point_size=point_size,
        alpha=alpha,
        output=output,
    )


if __name__ == "__main__":
    app()
