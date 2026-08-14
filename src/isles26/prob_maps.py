"""Foreground probability maps from nnU-Net softmax, on the native grid.

nnU-Net trained/predicted with ``--npz`` writes a ``{case}.npz`` of class
softmax next to each ``{case}.nii.gz`` prediction, both already reverted to the
input's original geometry and in SimpleITK array order. This extracts the
foreground channel as a native-grid float map -- the soft input PR-AUC needs,
and the probability output the submission must produce.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def load_foreground_softmax(npz_path) -> np.ndarray:
    """Foreground-class softmax array ``[Z, Y, X]`` from an nnU-Net ``.npz``."""
    with np.load(npz_path) as data:
        arr = data["probabilities"] if "probabilities" in data else data[data.files[0]]
    # [num_classes, Z, Y, X]; binary task -> foreground is class 1.
    return np.asarray(arr[1], dtype=np.float32)


def npz_to_prob_image(npz_path, reference: sitk.Image) -> sitk.Image:
    """Native-grid foreground-probability image aligned to ``reference``.

    ``reference`` is the matching prediction/GT mask (native geometry). nnU-Net
    stores the softmax at the original shape in SimpleITK array order, so the
    foreground channel drops straight onto the reference grid. A shape mismatch
    means the ``.npz`` is not at native geometry -- fail loudly rather than write
    a misaligned map (re-predict with ``--save_probabilities`` if this trips).
    """
    fg = load_foreground_softmax(npz_path)
    ref_shape = sitk.GetArrayViewFromImage(reference).shape
    if fg.shape != ref_shape:
        raise ValueError(
            f"softmax shape {fg.shape} != reference {ref_shape}: the .npz is not at "
            f"native geometry. Re-predict with `nnUNetv2_predict --save_probabilities`, "
            f"or resample using the case's .pkl properties."
        )
    prob = sitk.GetImageFromArray(fg)
    prob.CopyInformation(reference)  # spacing / origin / direction
    return prob
