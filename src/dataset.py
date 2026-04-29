from pathlib import Path

import osmnx as ox
import pandas as pd
import typer

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"

app = typer.Typer(help="Download OSM points of interest for a city.")

TAGS = {
    "amenity": ["restaurant", "cafe", "pharmacy", "school", "bank", "atm", "hospital"],
    "shop": ["supermarket", "convenience"],
    "tourism": ["hotel"],
    "highway": ["bus_stop"],
    "railway": ["tram_stop"],
    "leisure": ["park"],
}

PRIORITY_KEYS = ["amenity", "shop", "tourism", "highway", "railway", "leisure"]


def get_feature_type(row: pd.Series) -> str | None:
    """Return a single ``"key=value"`` label for an OSM feature row.

    OSM features can carry several relevant tags at once (for example a
    building tagged with both ``amenity=restaurant`` and ``tourism=hotel``).
    This function walks the keys in ``PRIORITY_KEYS`` in order and returns
    the first one whose value is present (not ``NaN``/``None``), so each
    feature ends up with exactly one categorical label.

    Args:
        row: A row from the OSM ``GeoDataFrame``; expected to expose the
            tag columns listed in ``PRIORITY_KEYS``.

    Returns:
        A string of the form ``"<key>=<value>"`` (e.g. ``"amenity=cafe"``),
        or ``None`` if the row has no value for any of the priority keys.
    """
    for key in PRIORITY_KEYS:
        val = row.get(key)
        if pd.notna(val):
            return f"{key}={val}"
    return None


def slugify(name: str) -> str:
    """Turn a free-form name into a filesystem-friendly slug.

    Lowercases the input, keeps only ASCII alphanumeric characters, and
    collapses any run of other characters (spaces, punctuation, accents,
    etc.) into a single underscore. Leading and trailing underscores are
    stripped so the result never starts or ends with one.

    Examples:
        >>> slugify("Warsaw")
        'warsaw'
        >>> slugify("New York")
        'new_york'
        >>> slugify("  Stoke-on-Trent!! ")
        'stoke_on_trent'

    Args:
        name: The original name (typically a city name).

    Returns:
        A slug suitable for use as a filename component.
    """
    chars: list[str] = []
    prev_underscore = False
    for ch in name.lower():
        if ch.isalnum():
            chars.append(ch)
            prev_underscore = False
        elif not prev_underscore:
            chars.append("_")
            prev_underscore = True
    return "".join(chars).strip("_")


def build_events(place: str) -> tuple[pd.DataFrame, dict]:
    info: dict = {}

    gdf = ox.features_from_place(place, TAGS)
    info["raw_count"] = len(gdf)
    info["geom_types"] = gdf.geom_type.value_counts()

    # Auto-pick an appropriate projected CRS (UTM zone) so distances are in meters
    # regardless of the country.
    gdf = ox.projection.project_gdf(gdf)
    info["crs"] = gdf.crs

    gdf["point"] = gdf.geometry.representative_point()
    gdf["x"] = gdf["point"].x
    gdf["y"] = gdf["point"].y
    info["x_min"], info["x_max"] = gdf["x"].min(), gdf["x"].max()
    info["y_min"], info["y_max"] = gdf["y"].min(), gdf["y"].max()

    gdf["feature_type"] = gdf.apply(get_feature_type, axis=1)
    info["unmatched"] = int(gdf["feature_type"].isna().sum())

    events = gdf[["feature_type", "x", "y"]].dropna().reset_index(drop=True)
    events["instance_id"] = events.index
    events = events[["instance_id", "feature_type", "x", "y"]]

    info["duplicate_coords"] = int(events.duplicated(subset=["x", "y"]).sum())
    return events, info


def print_stats(events: pd.DataFrame, info: dict) -> None:
    typer.echo(f"\nRaw features fetched: {info['raw_count']}")
    typer.echo("Geometry types:")
    typer.echo(info["geom_types"].to_string())

    typer.echo(f"\nProjected CRS: {info['crs']}")
    typer.echo("Coordinate ranges (meters):")
    typer.echo(
        f"  x: {info['x_min']:.2f} .. {info['x_max']:.2f}  "
        f"(span {info['x_max'] - info['x_min']:.2f} m)"
    )
    typer.echo(
        f"  y: {info['y_min']:.2f} .. {info['y_max']:.2f}  "
        f"(span {info['y_max'] - info['y_min']:.2f} m)"
    )

    typer.echo(f"\nRows without a matched feature_type (dropped): {info['unmatched']}")
    typer.echo(f"Final events: {len(events)}")
    typer.echo(f"Distinct feature types: {events['feature_type'].nunique()}")

    typer.echo("\nCounts per feature_type:")
    typer.echo(events["feature_type"].value_counts().to_string())

    typer.echo("\nCounts per top-level key:")
    typer.echo(events["feature_type"].str.split("=").str[0].value_counts().to_string())

    typer.echo("\nSample rows:")
    typer.echo(events.head(10).to_string(index=False))

    typer.echo(
        f"\nRows sharing exact (x,y) with another row: {info['duplicate_coords']}"
    )


@app.command()
def download(
    city: str = typer.Argument(..., help="City name, e.g. 'Warsaw'."),
    country: str = typer.Argument(..., help="Country name, e.g. 'Poland'."),
    stats: bool = typer.Option(
        False,
        "--stats",
        "-s",
        help="Print statistics about the downloaded dataset.",
    ),
    output_dir: Path = typer.Option(
        DATA_DIR,
        "--output-dir",
        "-o",
        help="Directory to write the CSV into.",
    ),
) -> None:
    """Download OSM points of interest for a city and save them as a flat CSV."""
    place = f"{city}, {country}"
    typer.echo(f"Fetching OSM features for: {place}")

    events, info = build_events(place)

    if len(events) == 0:
        typer.echo("No features matched the requested tags.", err=True)
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{slugify(city)}_osm_events.csv"
    events.to_csv(out_path, index=False)
    typer.echo(f"Saved {len(events)} rows to {out_path}")

    if stats:
        print_stats(events, info)


if __name__ == "__main__":
    app()
