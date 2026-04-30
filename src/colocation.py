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
