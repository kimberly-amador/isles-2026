"""Aggregate per-case metrics and rank candidate models.

``aggregate`` reduces per-case records to overall means. ``rank_aggregate``
mirrors the challenge's "rank then aggregate": per case, rank the models on each
scored metric, mean the per-metric ranks -> case rank, then mean across cases.
Lower final score = better.

The official metrics score empty-GT cases rather than dropping them (PR-AUC ->
1.0/0.0, Dice/F1 -> 1.0), so a NaN metric is now rare -- only the sklearn
fallback in ``compute_pr_auc``, or PR-AUC on a run without probability maps. The
means and ranks stay NaN-aware so such a value is excluded rather than poisoning
the aggregate.
"""
from __future__ import annotations

import numpy as np

# scored metric -> is-higher-better. These five drive the ranking.
SCORED_METRICS: dict[str, bool] = {
    "dice": True,
    "abs_volume_diff_ml": False,
    "pr_auc": True,
    "detection_f1": True,
    "abs_count_diff": False,
}


def aggregate(per_case: list[dict]) -> dict:
    """Overall mean of each scored metric across cases (NaN-aware)."""
    out: dict[str, float] = {"n_cases": len(per_case)}
    for m in SCORED_METRICS:
        vals = np.array([c[m] for c in per_case], dtype=float)
        out[f"{m}_mean"] = float(np.nanmean(vals)) if vals.size and not np.isnan(vals).all() else float("nan")
    return out


def _avg_ranks(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """1-based average ranks (1 = best). NaN inputs stay NaN and are not ranked."""
    ranks = np.full(values.shape, np.nan)
    finite = np.flatnonzero(~np.isnan(values))
    if finite.size == 0:
        return ranks
    v = values[finite]
    order = -v if higher_is_better else v
    sorter = np.argsort(order, kind="mergesort")
    sv = order[sorter]
    r = np.empty(v.shape)
    i = 0
    while i < sv.size:  # average ties so tied models share a rank
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        r[sorter[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks[finite] = r
    return ranks


def rank_aggregate(models: dict[str, list[dict]]) -> list[tuple[str, float]]:
    """Rank candidate models the way the challenge ranks teams.

    ``models`` maps model name -> its per-case records (each carrying a ``case``
    id and the scored metrics). Only cases present for *every* model are used.
    Returns ``[(model, mean_rank), ...]`` sorted best-first.
    """
    names = list(models)
    by_case = {n: {c["case"]: c for c in models[n]} for n in names}
    shared = sorted(set.intersection(*(set(d) for d in by_case.values())))
    if not shared:
        raise ValueError("models share no common cases to rank on")

    case_ranks: dict[str, list[float]] = {n: [] for n in names}
    for case in shared:
        per_metric = [
            _avg_ranks(np.array([by_case[n][case][m] for n in names], dtype=float), hib)
            for m, hib in SCORED_METRICS.items()
        ]
        mean_rank = np.nanmean(np.vstack(per_metric), axis=0)  # over the 5 metrics, per model
        for k, n in enumerate(names):
            case_ranks[n].append(float(mean_rank[k]))

    result = [(n, float(np.mean(case_ranks[n]))) for n in names]
    result.sort(key=lambda t: t[1])
    return result
