"""Binary lesion-mask postprocessing (native space, SimpleITK).

Removes unconvincing connected components to lift lesion-level Detection-F1 and
count-difference; optionally fills holes. Keeps ALL components that pass the
filters -- NOT nnU-Net's "keep largest", which would delete secondary lesions
(ISLES is majority multi-lesion). 26-connectivity matches panoptica's instance
approximation used in scoring.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def postprocess_mask(
    mask: sitk.Image,
    min_lesion_ml: float = 0.0,
    fill_holes: bool = False,
    prob: sitk.Image | None = None,
    min_component_prob: float = 0.0,
    close_mm: float = 0.0,
) -> sitk.Image:
    """Binarized mask with unconvincing connected components removed.

    A component is dropped if its volume is below ``min_lesion_ml`` or its peak
    probability is below ``min_component_prob``. The probability criterion needs
    ``prob`` (the soft map, on the mask's grid) and is skipped without it --
    volume alone cannot separate a confident small lesion from a low-confidence
    speck, which is why the two thresholds are tuned jointly.

    ``close_mm`` applies a binary closing before components are labelled, so a
    lesion predicted as several near-touching fragments is scored as one object.
    Detection-F1 matches at IoU >= 0.25, where fragments can each fall below the
    threshold and none of them match.

    With both thresholds at 0 and ``fill_holes=False`` this returns a binarized
    copy and drops nothing.
    """
    m = sitk.Cast(sitk.NotEqual(mask, 0), sitk.sitkUInt8)
    if close_mm > 0:
        # Radius in mm converted per axis: spacing spans 0.46-6 mm across the
        # cohort, so a voxel radius would be a different physical operation on
        # every case. An axis coarser than the radius rounds to 0 and is left
        # alone rather than being closed by a whole slice.
        radius = [int(round(close_mm / s)) for s in m.GetSpacing()]
        if any(radius):
            m = sitk.BinaryMorphologicalClosing(m, radius, sitk.sitkBall)
    if fill_holes:
        m = sitk.BinaryFillhole(m)

    use_prob = prob is not None and min_component_prob > 0
    if min_lesion_ml <= 0 and not use_prob:
        return m

    vox_ml = float(np.prod(m.GetSpacing())) / 1000.0
    min_voxels = max(1, round(min_lesion_ml / vox_ml)) if min_lesion_ml > 0 else 0

    ccf = sitk.ConnectedComponentImageFilter()
    ccf.SetFullyConnected(True)  # 26-conn
    cc = ccf.Execute(m)

    stats = sitk.LabelStatisticsImageFilter()
    # Without a soft map only voxel counts are read, so the mask itself is a
    # valid stand-in for the intensity argument.
    stats.Execute(prob if use_prob else sitk.Cast(m, sitk.sitkFloat32), cc)
    drop = {
        int(lbl): 0
        for lbl in stats.GetLabels()
        if lbl != 0
        and (stats.GetCount(lbl) < min_voxels
             or (use_prob and stats.GetMaximum(lbl) < min_component_prob))
    }
    if drop:
        cc = sitk.ChangeLabel(cc, drop)
    return sitk.Cast(sitk.NotEqual(cc, 0), sitk.sitkUInt8)
