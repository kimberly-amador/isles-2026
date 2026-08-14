"""Per-case ISLES 2026 metrics — thin SimpleITK adapters over the official code.

The scoring itself lives in `_official.py`, a verbatim mirror of `eval_utils.py`
from https://github.com/ezequieldlrosa/isles26. This module only turns SimpleITK
images into the numpy arrays those functions expect and packs one per-case
record. The five scored metrics mirror the official outputs exactly:

    dice                 panoptica global binary Dice
    abs_volume_diff_ml   |vol(pred) - vol(gt)|, millilitres
    pr_auc               area under the PR curve of the soft map
    detection_f1         lesion-wise F1 (panoptica recognition quality),
                         instances matched at IoU >= 0.25
    abs_count_diff       |#instances(pred) - #instances(gt)|

Empty-GT handling follows the official code (empty_value=1.0): both-empty scores
1.0 for Dice/F1; empty-GT PR-AUC is 1.0 only when the soft map is perfectly flat,
else 0.0. Inputs are `sitk.Image` on a shared native voxel grid; array axis order
does not affect these voxel-wise / connected-component statistics.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

CaseMetrics = dict  # flat {name: value} record for one case


def voxel_volume_ml(image: sitk.Image) -> float:
    """Physical volume of one voxel, in millilitres."""
    return float(np.prod(image.GetSpacing())) / 1000.0


def evaluate_case(
    gt: sitk.Image, pred: sitk.Image, prob: sitk.Image | None = None
) -> CaseMetrics:
    """The five official scored metrics (plus GT/pred volume) for one case.

    ``prob`` (soft foreground map on the GT grid) may be ``None`` -- then
    ``pr_auc`` is NaN and the other four are still scored, enough for a
    Dice/F1/volume/count comparison before probability maps are wired in.
    """
    # Imported lazily so aggregation/ranking stay usable without panoptica.
    from ._official import (
        compute_absolute_volume_difference,
        compute_dice_f1_instance_difference,
        compute_pr_auc,
    )

    g = sitk.GetArrayFromImage(gt)
    p = sitk.GetArrayFromImage(pred)
    vox_ml = voxel_volume_ml(gt)

    f1, count_diff, dice_score = compute_dice_f1_instance_difference(g, p)
    pr_auc = (
        float(compute_pr_auc(g, sitk.GetArrayFromImage(prob)))
        if prob is not None
        else float("nan")
    )
    return {
        "dice": float(dice_score),
        "abs_volume_diff_ml": float(compute_absolute_volume_difference(g, p, np.array(vox_ml))),
        "pr_auc": pr_auc,
        "detection_f1": float(f1),
        "abs_count_diff": int(count_diff),
        # supporting fields for breakdowns / debugging
        "gt_volume_ml": float(np.count_nonzero(g)) * vox_ml,
        "pred_volume_ml": float(np.count_nonzero(p)) * vox_ml,
    }
