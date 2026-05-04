# Spatial Co-location Pattern Mining (Huang-Shekhar-Xiong, 2004)

This repository contains a Python implementation of the event-centric co-location mining algorithm from:

Y. Huang, S. Shekhar, H. Xiong, *Discovering Co-location Patterns from Spatial Data Sets: A General Approach*, IEEE TKDE, 2004.

The implementation mines prevalent spatial co-locations using:
- Euclidean neighbor relation `R` with threshold distance `d`
- Participation Index (PI) as prevalence measure
- Apriori-style candidate generation with anti-monotonic pruning
- Geometric size-2 table-instance generation (KD-tree)
- Combinatorial joins for larger co-locations
- Co-location rule generation with conditional probability threshold

## Project layout

```text
src/
  colocation.py   # core algorithm
  dataset.py      # OpenStreetMap dataset download + CSV export
  visualize.py    # scatter visualization of dataset points
  main.py         # CLI to run mining and export results
  plot.py         # summary and spatial result plots
data/             # downloaded datasets (CSV)
results/          # mined outputs (CSV + PNG)
tests/            # unit and smoke tests
```

## Requirements

- Python 3.13+
- `uv` package manager

Install dependencies:

```bash
uv sync --dev
```

## 1) Download datasets

Datasets are built from OpenStreetMap POIs for a city/country pair and saved as event CSV files with columns:

`instance_id, feature_type, x, y`

Example:

```bash
uv run dataset Warsaw Poland --stats
uv run dataset Prague Czechia --stats
```

By default this writes:

- `data/warsaw_osm_events.csv`
- `data/prague_osm_events.csv`

You can choose a custom destination with `--output-dir`.

## 2) Visualize downloaded datasets

Quick visualization (interactive window):

```bash
uv run visualize data/warsaw_osm_events.csv
```

Save to an image file:

```bash
uv run visualize data/warsaw_osm_events.csv --color-by feature_type --output results/warsaw_events.png
```

Common filters:
- `--key amenity --key shop`
- `--feature-type amenity=restaurant`
- `--bbox xmin,ymin,xmax,ymax`
- `--sample 5000`

## 3) Run co-location mining and generate results

Basic run:

```bash
uv run main data/warsaw_osm_events.csv
```

Run with explicit thresholds, save CSV outputs, and generate plots:

```bash
uv run main data/warsaw_osm_events.csv --distance 100 --prevalence 0.3 --conditional 0.5 --min-count 20 --top-rules 30 --output-dir results --save --plot
```

## Output artifacts

For input `data/warsaw_osm_events.csv`, the pipeline writes:

- `results/warsaw_prevalent_colocations.csv`
- `results/warsaw_rules.csv`
- `results/warsaw_summary.png`
- `results/warsaw_spatial_colocations.png`

### CSV content

- `*_prevalent_colocations.csv`: prevalent co-locations, PI, per-feature participation ratios, table-instance counts
- `*_rules.csv`: rules `antecedent => consequent`, rule size, prevalence (PI), conditional probability

### PNG content

- `*_summary.png`: prevalence by size, top co-locations, rule quality scatter, top rules
- `*_spatial_colocations.png`: spatial overlays of selected prevalent co-locations

## Reproduce included example outputs

The repository already contains example outputs for Warsaw and Prague in `results/`.
To regenerate them:

```bash
uv run main data/warsaw_osm_events.csv --plot
uv run main data/prague_osm_events.csv --plot
```

## Testing

Run the test suite:

```bash
uv run pytest
```

`tests/test_smoketest.py` includes a regression check reproducing Fig. 2 PI values from the paper setup.
