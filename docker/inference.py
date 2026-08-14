"""ISLES 2026 algorithm container — nnU-Net inference for Grand Challenge.

Implements the two hooks the template's `app.py` calls:

    init_model()  once at server startup, before /health returns 200
    run(model)    once per case, reading /input and writing /output

Interface `interf0`:
    in   /input/images/t1-brain-mri/  (.mha at runtime)
         /input/stroke-metadata.json  (present, deliberately unused)
    out  /output/images/stroke-lesion-segmentation/output.mha   uint8 mask
         /output/images/lesion-probability-map/output.mha       float32 in [0, 1]

Contract and limits: `docs/submission_interface.md`.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from postprocessing import postprocess_mask

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
# Grand Challenge extracts the uploaded model tarball here, read-only. It must
# contain an nnU-Net trained-model folder: dataset.json, plans.json, fold_N/.
MODEL_PATH = Path("/opt/ml/model")

# --- Deployment knobs -------------------------------------------------------
# These match the configuration our CV scores describe. nnU-Net's
# end-of-training validation runs WITH mirroring, so every reported number
# (Dice 0.6582, Det-F1 0.6091, PR-AUC 0.7594) is a TTA-enabled measurement --
# turning it off here would ship something never scored. Full 8x mirroring on
# the worst-case geometry is ~421 s against a 420 s/case limit, so if this stays
# on, the algorithm needs a GPU. See docs/roadmap.md -> Submission.
USE_MIRRORING = True
# The `-f all` retrain: same recipe as the CV, fitted to all 1450 cases. Loads
# fold_all/. `checkpoint_best` is selected on in-sample validation here (-f all
# validates on its own training set), so only `final` carries signal.
FOLDS = ("all",)
CHECKPOINT = "checkpoint_final.pth"
# Tuned on the 1450-case CV; best floor in every dataset x trainer sweep.
PP_MIN_ML = 0.05


def init_model():
    """Load the predictor once; reused across every /invoke."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=USE_MIRRORING,
        # Keeping the sliding window on-device is a CUDA-only optimisation.
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_PATH), use_folds=FOLDS, checkpoint_name=CHECKPOINT
    )
    print(f"loaded folds {FOLDS} from {MODEL_PATH}, mirroring={USE_MIRRORING}", flush=True)
    return predictor


def run(model):
    """Called per /invoke. Dispatches on the sockets this case provides."""
    handler = {
        ("stroke-metadata", "t1-brain-mri"): interf0_handler,
    }[get_interface_key()]
    return handler(model)


def interf0_handler(predictor):
    t1_path = find_input_image(INPUT_PATH / "images" / "t1-brain-mri")
    # Read twice, on purpose: nnU-Net wants its own (array, properties) pair, and
    # the sitk.Image is what stamps geometry back onto the outputs.
    reference = sitk.ReadImage(str(t1_path))
    data, properties = SimpleITKIO().read_images([str(t1_path)])

    # nnU-Net resamples its own output back to the input grid, which is what
    # satisfies the challenge's native-space contract on this track.
    segmentation, probabilities = predictor.predict_single_npy_array(
        data, properties, None, None, True
    )

    mask = as_image(segmentation.astype(np.uint8), reference)
    # Channel 1 of the softmax is the lesion class; channel 0 is background.
    prob = as_image(probabilities[1].astype(np.float32), reference)

    mask = postprocess_mask(mask, PP_MIN_ML)
    # An empty mask means "no lesion". Empty-GT PR-AUC scores 1.0 only for a
    # perfectly flat soft map and ~0 on incidental variation, so zero it --
    # unconditionally, unlike the eval harness where it is a sweep knob.
    if not sitk.GetArrayViewFromImage(mask).any():
        prob = sitk.Multiply(prob, 0.0)

    write_image(OUTPUT_PATH / "images" / "stroke-lesion-segmentation", mask)
    write_image(OUTPUT_PATH / "images" / "lesion-probability-map", prob)
    return 0


def get_interface_key():
    """Sorted socket slugs for this case, from the system-generated inputs.json."""
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["socket"]["slug"] for sv in inputs))


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def find_input_image(location: Path) -> Path:
    """Grand Challenge converts uploads to .mha; the others are for local runs."""
    for pattern in ("*.mha", "*.nii.gz", "*.nii"):
        hits = sorted(glob.glob(str(location / pattern)))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"No image found in {location}")


def as_image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Wrap a (z, y, x) array on the reference's exact grid."""
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def write_image(directory: Path, image: sitk.Image) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(directory / "output.mha"), useCompression=True)


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
