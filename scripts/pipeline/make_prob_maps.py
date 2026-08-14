"""Write native-grid foreground probability maps from an nnU-Net prediction dir.

nnU-Net's validation (or predict) folder holds ``{case}.npz`` (class softmax)
next to ``{case}.nii.gz`` (the mask). This writes ``{case}.nii.gz`` float
probability maps to ``--out``, ready for ``evaluate.py --prob-dir`` (PR-AUC) and
usable as the submission's soft output.

Usage::

    python scripts/pipeline/make_prob_maps.py \\
        --pred-dir .../fold_0/validation --out .../fold_0/prob
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from isles26.prob_maps import npz_to_prob_image


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", required=True, help="nnU-Net dir with {case}.npz + {case}.nii.gz")
    ap.add_argument("--out", required=True, help="output dir for float probability maps")
    args = ap.parse_args(argv)

    pred_dir, out = Path(args.pred_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    npzs = sorted(pred_dir.glob("*.npz"))
    if not npzs:
        raise SystemExit(f"No .npz in {pred_dir} — train/predict with --npz / --save_probabilities")

    for i, npz in enumerate(npzs, 1):
        case = npz.stem
        ref = pred_dir / f"{case}.nii.gz"
        if not ref.exists():
            print(f"  [skip] {case}: no matching mask {ref.name}", file=sys.stderr)
            continue
        prob = npz_to_prob_image(npz, sitk.ReadImage(str(ref)))
        sitk.WriteImage(prob, str(out / f"{case}.nii.gz"), useCompression=True)
        if i % 25 == 0 or i == len(npzs):
            print(f"  {i}/{len(npzs)} prob maps written (last={case})", file=sys.stderr)

    print(f"\nWrote prob maps to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
