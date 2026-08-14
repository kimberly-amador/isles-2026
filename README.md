# ISLES 2026 — T1w stroke lesion segmentation

Code accompanying our submission to the [ISLES 2026 Challenge](https://isles-26.grand-challenge.org/isles-26/):
binary ischemic stroke lesion segmentation from a single native-space
T1-weighted MRI volume.

The method is nnU-Net v2 with the **medium residual-encoder preset (ResEnc M)**
and a **connected-component volume filter** applied to the predicted mask. This
repository contains the parts that are ours — the postprocessing, the exact
cross-validation folds, the scoring harness, and the configuration needed to
reproduce training. nnU-Net itself is not vendored; it is pinned by version.

## What is here

| path | what it is |
| --- | --- |
| `docker/` | the submitted algorithm container |
| `src/isles26/postprocessing.py` | the connected-component filter (the method) |
| `src/isles26/eval/` | scoring harness: the five challenge metrics |
| `scripts/pipeline/evaluate.py` | CLI to score predictions and rank configurations |
| `scripts/pipeline/make_prob_maps.py` | nnU-Net `.npz` softmax → probability maps for PR-AUC |
| `scripts/paper/stats_postprocessing.py` | paired significance tests for the volume threshold |
| `reports/splits/` | the exact 5-fold split, and how it was stratified |
| `reports/eval/` | per-case scores behind the reported tables |
| `nnunet/` | the nnU-Net fingerprint, plans, and dataset descriptor that pin both architectures |

## Reproducing training

Build the raw dataset in nnU-Net format (one T1w volume and one binary lesion
mask per case, `channel_names: {0: T1w}`, labels `{background: 0, lesion: 1}` —
see `nnunet/dataset.json`), then:

```bash
conda env create -f environment.yml && conda activate isles26

D="$nnUNet_preprocessed/Dataset026_ISLES26"

# 1. Fingerprint the raw data, then substitute ours so planning cannot drift.
nnUNetv2_extract_fingerprint -d 026
cp nnunet/dataset_fingerprint.json "$D/"

# 2. Use our plans rather than re-deriving them from your copy of the data.
cp nnunet/nnUNetPlans.json nnunet/nnUNetResEncUNetMPlans.json "$D/"

# 3. Preprocess against each plans file; ResEnc M writes its own cache.
nnUNetv2_preprocess -d 026 -plans_name nnUNetPlans            -c 3d_fullres
nnUNetv2_preprocess -d 026 -plans_name nnUNetResEncUNetMPlans -c 3d_fullres

# 4. Use our folds; nnU-Net would otherwise generate its own on first training.
cp reports/splits/splits_final.json "$D/splits_final.json"

# 5. Train each fold, FOLD in 0..4.
nnUNetv2_train 026 3d_fullres FOLD -p nnUNetResEncUNetMPlans --npz   # ResEnc M
nnUNetv2_train 026 3d_fullres FOLD -p nnUNetPlans            --npz   # Default
```

**Plans.** `nnunet/nnUNetResEncUNetMPlans.json` is the configuration we submitted
— `ResidualEncoderUNet`, patch size 128³, batch size 2, target spacing 1.0 mm
isotropic. `nnunet/nnUNetPlans.json` is the Default 3D full-resolution baseline
it is compared against. Copying both in, rather than re-planning, guarantees an
identical architecture; nnU-Net resolves plans by filename, so keep the names as
they are.

**Fingerprint.** Both configurations were planned from the single shared
`nnunet/dataset_fingerprint.json`, which is what makes the comparison controlled:
the two presets differ in encoder topology and in nothing upstream of it.

**Splits.** `reports/splits/splits_final.json` is in nnU-Net's native format: a
list of five `{train, val}` objects over 1450 cases. Folds are grouped by
subject (no subject appears in both sides of a split) and stratified by
acquisition site × lesion-volume quartile × chronicity, seed 0;
`reports/splits/summary.txt` records the quartile edges and the stratum counts.

## The submitted algorithm

`docker/` is the container submitted to Grand Challenge, built on their ISLES'26
algorithm template. `inference.py` holds the prediction path; `app.py` is the
template's `invoke` server.

```bash
bash docker/do_build.sh
```

The build stages `src/isles26/postprocessing.py` into the context rather than
keeping a second copy, so the filter that ships is by construction the one the
cross-validation was scored with.

Model weights are **not** in this repository. The container expects them mounted
read-only at `/opt/ml/model`, holding `dataset.json`, `plans.json` and
`fold_all/checkpoint_final.pth`.

The submitted model is a single `nnUNetTrainer` run over all training data
(`-f all`), predicted with **mirroring test-time augmentation enabled**. Two
outputs are written per case, both on the input's native voxel grid: a binary
`uint8` mask and a float probability map.

Postprocessing, in order:

1. drop connected components smaller than **0.05 mL** (`postprocess_mask`);
2. if the resulting mask is empty, **flatten the probability map to a constant**.

Step 2 matters for scoring: the challenge assigns PR-AUC 1.0 on an empty
ground truth only when the probability map is perfectly flat, and 0.0 otherwise.

## Scoring

Scoring is CPU-only and has its own environment, with `panoptica` pinned —
panoptica 1.x and 2.x compute Dice differently, so scores from different major
versions are not comparable.

```bash
conda env create -f environment-eval.yml && conda activate isles26-eval

python scripts/pipeline/evaluate.py evaluate \
    --pred-dir PREDICTIONS --gt-dir LABELS --prob-dir PROBABILITY_MAPS \
    --pp-min-ml 0.05 --out OUT --name resencm

python scripts/pipeline/evaluate.py compare \
    a=OUT_A/per_case.csv b=OUT_B/per_case.csv
```

`evaluate` writes `per_case.csv` and a summary broken down by site and by lesion
size. `compare` implements the challenge's rank-then-aggregate scheme: rank the
configurations per case on each of the five metrics, average the ranks per case,
then average across cases.

Metrics follow the official implementation at
<https://github.com/ezequieldlrosa/isles26>: Dice, absolute volume difference
(mL), PR-AUC of the soft map, lesion-wise detection F1, and absolute lesion-count
difference, with lesions matched as connected components at IoU ≥ 0.25.

## Significance tests

`scripts/paper/stats_postprocessing.py` reproduces the paired tests for the
volume threshold from the per-case scores in `reports/eval/`, with no GPU and no
retraining:

```bash
conda activate isles26-eval && python scripts/paper/stats_postprocessing.py
```

## Data

Training data is ATLAS v3.0 as redistributed for ISLES 2026, available from the
challenge organizers. It is derived from ATLAS v2.0 pooled with additional sites.
No imaging data is included in this repository.

- Liew, S.-L. et al. A large, curated, open-source stroke neuroimaging dataset to
  improve lesion segmentation algorithms. *Scientific Data* **9**, 320 (2022).
- Absher, J. et al. The Stroke Outcome Optimization Project: acute ischemic
  strokes from a comprehensive stroke center. *Scientific Data* **11** (2024).
