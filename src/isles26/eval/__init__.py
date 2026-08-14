"""ISLES 2026 evaluation harness — official metrics + rank aggregation.

``metrics.evaluate_case`` scores one case by wrapping the challenge's official
code (``_official.py``, a verbatim mirror of the ISLES'26 ``eval_utils.py``);
``aggregate.py`` does the "rank then aggregate" model comparison. Scoring needs
``panoptica`` installed; the aggregation/ranking path does not.
"""
from .aggregate import SCORED_METRICS, aggregate, rank_aggregate
from .metrics import evaluate_case, voxel_volume_ml

__all__ = [
    "evaluate_case",
    "voxel_volume_ml",
    "aggregate",
    "rank_aggregate",
    "SCORED_METRICS",
]
