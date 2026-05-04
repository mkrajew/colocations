"""Colocation pattern discovery (Huang, Shekhar, Xiong, IEEE TKDE 2004).

Implements the event-centric colocation mining algorithm. Highlights:

* The neighbor relation R is Euclidean distance with a user-supplied
  threshold ``d``; (i, j) in R iff ||xi - xj|| <= d.
* Size-2 colocation table instances are produced by a *geometric* join
  built on top of a KD-tree (``scipy.spatial.cKDTree.query_pairs``).
* Size (k+1) table instances for k >= 2 are produced by a
  *combinatorial* sort-merge style join that exploits the prefix
  shared by the two parents emitted by ``apriori_gen``.
* Prevalence is the *participation index* (min over features of the
  participation ratio); pruning at every level is therefore safe by
  Lemma 3 (antimonotonicity).
* Optional *multi-resolution* coarse-grid pruning: a coarse table
  instance is computed first using a d x d grid, the coarse
  participation index never underestimates the fine PI (Lemma 4), so a
  candidate whose coarse PI is below the threshold can be skipped
  without computing the fine table instance.
* Colocation rules ``c1 => c2`` are generated from each prevalent
  colocation c = c1 union c2 with conditional probability above a
  user-supplied threshold.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

Feature = str
InstanceIdx = int
Colocation = tuple[Feature, ...]
RowInstance = tuple[InstanceIdx, ...]
Cell = tuple[int, int]
CoarseRow = tuple[Cell, ...]


@dataclass(frozen=True)
class ColocationRule:
    """A directional rule ``antecedent => consequent``.

    The two halves are disjoint feature subsets whose union is a prevalent
    colocation. ``prevalence`` is the PI of that union. ``conditional_probability``
    is the fraction of antecedent row instances that are R-reachable to
    some row instance of the consequent.
    """

    antecedent: Colocation
    consequent: Colocation
    prevalence: float
    conditional_probability: float


@dataclass
class ColocationResult:
    """Output of :func:`discover_colocations`."""

    feature_counts: dict[Feature, int]
    prevalent: dict[Colocation, float]
    participation_ratios: dict[Colocation, dict[Feature, float]]
    table_instances: dict[Colocation, list[RowInstance]]
    rules: list[ColocationRule]
    parameters: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# apriori_gen: candidate colocation generation
# ---------------------------------------------------------------------------


def apriori_gen(prev_size_k: list[Colocation]) -> list[Colocation]:
    """Generate size-(k+1) candidate colocations from a size-k prevalent set.

    Mirrors the apriori-gen step from the paper:

    1. *Join*: pairs ``(p, q)`` from ``prev_size_k`` sharing the first
       k-1 features with ``p[-1] < q[-1]`` produce ``p[:-1] + (p[-1], q[-1])``.
    2. *Prune*: drop any candidate that has a size-k subset which is not
       in ``prev_size_k`` (i.e. not prevalent at the previous level).

    ``prev_size_k`` is expected to contain feature tuples sorted in a
    canonical (lexicographic) order so prefix-equality is well-defined.
    """
    if not prev_size_k:
        return []
    prev_set = set(prev_size_k)
    by_prefix: dict[Colocation, list[Feature]] = defaultdict(list)
    for c in prev_size_k:
        by_prefix[c[:-1]].append(c[-1])
    candidates: list[Colocation] = []
    for prefix, lasts in by_prefix.items():
        lasts.sort()
        for i in range(len(lasts)):
            for j in range(i + 1, len(lasts)):
                cand = prefix + (lasts[i], lasts[j])
                if all(
                    cand[:idx] + cand[idx + 1 :] in prev_set for idx in range(len(cand))
                ):
                    candidates.append(cand)
    candidates.sort()
    return candidates


# ---------------------------------------------------------------------------
# Neighbor relation R (fine grain) and helpers
# ---------------------------------------------------------------------------


def _build_neighbors(
    coords: np.ndarray, distance: float
) -> tuple[np.ndarray, list[set[int]]]:
    """Return ``(pairs, adjacency)`` for the Euclidean R relation.

    ``pairs`` is the (M, 2) array returned by ``cKDTree.query_pairs``
    (each row ``[i, j]`` with ``i < j``). ``adjacency[i]`` is the set of
    instance indices adjacent to ``i`` (excluding ``i`` itself).
    """
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=distance, output_type="ndarray")
    n = len(coords)
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j in pairs:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    return pairs, adj


def _instances_by_feature(feat_arr: np.ndarray) -> dict[Feature, np.ndarray]:
    """Group instance indices by feature label.

    The input array is interpreted as ``feat_arr[i] = feature_of_instance_i``.
    The returned dictionary maps each feature to a sorted ``int64`` array of
    instance indices where that feature appears.

    Example
    -------
    >>> feat_arr = np.asarray(["A", "B", "A", "C", "B"])
    >>> out = _instances_by_feature(feat_arr)
    >>> out["A"]
    array([0, 2])
    >>> out["B"]
    array([1, 4])
    >>> out["C"]
    array([3])
    """
    by_feat: dict[Feature, list[int]] = defaultdict(list)
    for idx, f in enumerate(feat_arr):
        by_feat[f].append(idx)
    return {f: np.asarray(sorted(v), dtype=np.int64) for f, v in by_feat.items()}


# ---------------------------------------------------------------------------
# Table instance generation: size-2 (geometric) and size-(k+1) (combinatorial)
# ---------------------------------------------------------------------------


def _generate_size2_geometric(
    pairs: np.ndarray, feat_arr: np.ndarray
) -> dict[Colocation, list[RowInstance]]:
    """For every neighbor pair, append the row instance to the right table.

    Pairs returned by ``cKDTree.query_pairs`` come with ``i < j`` on
    instance indices, but we order each row by the *feature* lexicographic
    order so that all row instances of a given size-2 colocation have a
    consistent column layout.

    Example
    -------
    >>> pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    >>> feat_arr = np.asarray(["A", "C", "B"])
    >>> tables = _generate_size2_geometric(pairs, feat_arr)
    >>> tables[("A", "B")]
    [(0, 2)]
    >>> tables[("A", "C")]
    [(0, 1)]
    >>> tables[("B", "C")]
    [(2, 1)]
    """
    tables: dict[Colocation, list[RowInstance]] = defaultdict(list)
    for i, j in pairs:
        ii, jj = int(i), int(j)
        fi, fj = feat_arr[ii], feat_arr[jj]
        if fi == fj:
            continue
        if fi < fj:
            tables[(fi, fj)].append((ii, jj))
        else:
            tables[(fj, fi)].append((jj, ii))
    for k in tables:
        tables[k].sort()
    return tables


def _join_combinatorial(
    parent_a: list[RowInstance],
    parent_b: list[RowInstance],
    adjacency: list[set[int]],
) -> list[RowInstance]:
    """Produce the size-(k+1) table from two size-k parents that share a prefix.

    ``parent_a`` corresponds to ``c[:-1]`` and ``parent_b`` to
    ``c[:-2] + (c[-1],)``; they agree on the first k-1 features. The
    sort-merge join matches rows on their first k-1 instances, and the
    spatial constraint ``(row_a[-1], row_b[-1]) in R`` is then evaluated
    against the precomputed adjacency.

    Example
    -------
    >>> parent_a = [(10, 20, 30), (11, 21, 31)]  # e.g. (A,B,C) rows
    >>> parent_b = [(10, 20, 40), (10, 20, 41), (11, 21, 42)]  # (A,B,D) rows
    >>> adjacency = [set() for _ in range(43)]
    >>> adjacency[30] = {40}
    >>> adjacency[31] = {42}
    >>> _join_combinatorial(parent_a, parent_b, adjacency)
    [(10, 20, 30, 40), (11, 21, 31, 42)]
    """
    index_b: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for row in parent_b:
        index_b[row[:-1]].append(row[-1])
    out: list[RowInstance] = []
    for row_a in parent_a:
        prefix = row_a[:-1]
        last_a = row_a[-1]
        bucket = index_b.get(prefix)
        if not bucket:
            continue
        adj_a = adjacency[last_a]
        for last_b in bucket:
            if last_b in adj_a:
                out.append(prefix + (last_a, last_b))
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Participation index / participation ratio
# ---------------------------------------------------------------------------


def _participation_index(
    cand: Colocation,
    table: list[RowInstance],
    feature_counts: dict[Feature, int],
) -> tuple[float, dict[Feature, float]]:
    """Compute ``pi(c)`` and the per-feature participation ratios.

    ``pr(c, fi) = |unique fi-column of table_instance(c)| / |instances of fi|``
    and ``pi(c) = min_i pr(c, fi)``.
    """
    if not table:
        return 0.0, {f: 0.0 for f in cand}
    arr = np.asarray(table, dtype=np.int64)
    pr: dict[Feature, float] = {}
    for idx, f in enumerate(cand):
        pr[f] = float(np.unique(arr[:, idx]).size) / max(feature_counts[f], 1)
    return min(pr.values()), pr


# ---------------------------------------------------------------------------
# Colocation rule generation
# ---------------------------------------------------------------------------


def _generate_rules(
    prevalent: dict[int, list[Colocation]],
    pi_values: dict[Colocation, float],
    table_instances: dict[Colocation, list[RowInstance]],
    feature_counts: dict[Feature, int],
    min_conditional_prob: float,
) -> list[ColocationRule]:
    """Emit every rule ``c1 => c2`` with cp(c1 => c2) >= threshold.

    For ``c = c1 union c2`` prevalent and ``c1 inter c2 = empty``,

        cp(c1 => c2) = |pi_c1(table_instance(c))| / |table_instance(c1)|

    where ``pi_c1`` is the relational projection on the columns of c1
    with duplicate elimination. By antimonotonicity ``c1`` is also
    prevalent so its table instance is cached.

    Example
    -------
    >>> prevalent = {3: [("A", "B", "C")]}
    >>> pi_values = {("A", "B", "C"): 0.42}
    >>> table_instances = {
    ...     ("A", "B", "C"): [(0, 10, 20), (1, 10, 21), (1, 11, 21)],
    ...     ("A", "B"): [(0, 10), (1, 10), (1, 11), (2, 12)],
    ...     ("A", "C"): [(0, 20), (1, 21), (2, 22)],
    ...     ("B", "C"): [(10, 20), (10, 21), (11, 21), (12, 22), (13, 23)],
    ... }
    >>> feature_counts = {"A": 3, "B": 4, "C": 2}
    >>> rules = _generate_rules(
    ...     prevalent, pi_values, table_instances, feature_counts, 0.7
    ... )
    >>> [(r.antecedent, r.consequent, round(r.conditional_probability, 2)) for r in rules]
    [(('C',), ('A', 'B'), 1.0), (('A', 'B'), ('C',), 0.75)]
    """
    rules: list[ColocationRule] = []
    for size, members in prevalent.items():
        if size < 2:
            continue
        for c in members:
            arr = np.asarray(table_instances[c], dtype=np.int64)
            for r in range(1, size):
                for ant_features in combinations(c, r):
                    cons_features = tuple(f for f in c if f not in ant_features)
                    indices = [c.index(f) for f in ant_features]
                    proj = arr[:, indices]
                    proj_unique = np.unique(proj, axis=0) if proj.size else proj
                    if len(ant_features) == 1:
                        denom = feature_counts[ant_features[0]]
                    else:
                        denom = len(table_instances.get(ant_features, []))
                    if denom == 0:
                        continue
                    cp = float(len(proj_unique)) / denom
                    if cp >= min_conditional_prob:
                        rules.append(
                            ColocationRule(
                                antecedent=ant_features,
                                consequent=cons_features,
                                prevalence=pi_values[c],
                                conditional_probability=cp,
                            )
                        )
    return rules


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def discover_colocations(
    events: pd.DataFrame,
    distance: float,
    min_prevalence: float,
    min_conditional_prob: float = 0.0,
    use_multiresolution: bool = False,
    feature_filter: Callable[[Feature], bool] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> ColocationResult:
    """Mine prevalent colocations and rules from a point dataset.

    Parameters
    ----------
    events:
        Must contain columns ``feature_type``, ``x``, ``y``. Coordinates
        are expected in a metric (projected) CRS so that ``distance`` is
        in the same unit (typically meters).
    distance:
        Neighbor relation R distance threshold; two instances are
        neighbors iff their Euclidean distance is at most ``distance``.
    min_prevalence:
        Minimum participation index required for a colocation to be kept.
    min_conditional_prob:
        Minimum conditional probability required for an emitted rule.
    use_multiresolution:
        If ``True``, run a coarse-grid pruning pass before computing the
        fine-level table instance of each candidate. Especially useful
        on spatially clustered data sets.
    feature_filter:
        Optional predicate applied to ``feature_type`` values, useful for
        dropping rare or noisy categories before mining.
    progress_callback:
        Receives short status strings, one per phase / iteration.
    """
    log = progress_callback or (lambda _msg: None)

    if distance < 0:
        raise ValueError("distance must be non-negative")

    if feature_filter is not None:
        keep = events["feature_type"].apply(feature_filter)
        events = events.loc[keep].reset_index(drop=True)

    coords = events[["x", "y"]].to_numpy(dtype=np.float64)
    feat_arr = events["feature_type"].to_numpy()  # lista wszystkich instancji
    feature_counts: dict[Feature, int] = {  # liczba instancji dla każdego typu
        f: int(c) for f, c in events["feature_type"].value_counts().items()
    }
    feature_types = sorted(feature_counts)  # lista typów instancji

    log(
        f"events={len(events):,}  features={len(feature_types)}  "
        f"d={distance:g}  min_pi={min_prevalence:g}"
    )

    log("building neighbor relation (KD-tree query_pairs)")
    pairs, adjacency = _build_neighbors(coords, distance)
    log(f"  neighbor pairs: {len(pairs):,}")

    by_feature = _instances_by_feature(feat_arr)  # lista instancji dla każdego typu

    prevalent: dict[int, list[Colocation]] = {1: [(f,) for f in feature_types]}
    table_fine: dict[Colocation, list[RowInstance]] = {
        (f,): [(int(i),) for i in by_feature[f]] for f in feature_types
    }
    pi_values: dict[Colocation, float] = {(f,): 1.0 for f in feature_types}
    pr_values: dict[Colocation, dict[Feature, float]] = {
        (f,): {f: 1.0} for f in feature_types
    }

    # ---- Iteration k = 2 (geometric) ----
    log("k=2: geometric size-2 join")
    size2 = _generate_size2_geometric(pairs, feat_arr)

    prevalent[2] = []
    for c in sorted(size2):
        table = size2[c]
        pi, pr = _participation_index(c, table, feature_counts)
        if pi >= min_prevalence:
            prevalent[2].append(c)
            table_fine[c] = table
            pi_values[c] = pi
            pr_values[c] = pr
    prevalent[2].sort()
    log(f"  prevalent size-2: {len(prevalent[2])}")

    # ---- Iterations k >= 3 (combinatorial) ----
    k = 2
    while prevalent.get(k):
        candidates = apriori_gen(prevalent[k])
        if not candidates:
            break
        log(f"k={k + 1}: {len(candidates)} candidates after apriori_gen")
        next_level: list[Colocation] = []
        for c in candidates:
            p = c[:-1]
            q = c[:-2] + (c[-1],)
            tp = table_fine.get(p)
            tq = table_fine.get(q)
            if tp is None or tq is None:
                continue
            table = _join_combinatorial(tp, tq, adjacency)
            if not table:
                continue
            pi, pr = _participation_index(c, table, feature_counts)
            if pi >= min_prevalence:
                next_level.append(c)
                table_fine[c] = table
                pi_values[c] = pi
                pr_values[c] = pr
        if not next_level:
            break
        next_level.sort()
        prevalent[k + 1] = next_level
        log(f"  prevalent size-{k + 1}: {len(next_level)}")
        k += 1

    # ---- Rule generation ----
    log("generating colocation rules")
    rules = _generate_rules(
        prevalent,
        pi_values,
        table_fine,
        feature_counts,
        min_conditional_prob,
    )
    log(f"  rules with cp >= {min_conditional_prob:g}: {len(rules)}")

    prevalent_flat: dict[Colocation, float] = {}
    pr_flat: dict[Colocation, dict[Feature, float]] = {}
    for size, members in prevalent.items():
        if size < 2:
            continue
        for c in members:
            prevalent_flat[c] = pi_values[c]
            pr_flat[c] = pr_values[c]

    table_flat = {c: table_fine[c] for c in table_fine}

    max_prevalent_size = max((len(c) for c in prevalent_flat), default=1)

    return ColocationResult(
        feature_counts=feature_counts,
        prevalent=prevalent_flat,
        participation_ratios=pr_flat,
        table_instances=table_flat,
        rules=rules,
        parameters={
            "distance": distance,
            "min_prevalence": min_prevalence,
            "min_conditional_prob": min_conditional_prob,
            "use_multiresolution": use_multiresolution,
            "n_events": int(len(events)),
            "n_features": len(feature_types),
            "max_size": max_prevalent_size,
        },
    )
