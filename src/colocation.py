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

import numpy as np
from scipy.spatial import cKDTree

Feature = str
InstanceIdx = int
Colocation = tuple[Feature, ...]
RowInstance = tuple[InstanceIdx, ...]
Cell = tuple[int, int]
CoarseRow = tuple[Cell, ...]


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
