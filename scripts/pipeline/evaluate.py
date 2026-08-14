"""Score predictions against ground truth with the 5 ISLES 2026 metrics.

Two modes:

  evaluate  Compute per-case metrics for a directory of predicted masks against
            the native-space GT masks (e.g. nnU-Net's ``labelsTr``). Writes
            ``per_case.csv`` + ``summary.txt`` (overall + by-site + by-size),
            with breakdowns derived from the case id and GT volume -- no extra
            metadata file needed. Soft probability maps are optional; without
            them PR-AUC is NaN and the other four metrics are still scored.
            ``--threshold`` re-derives the mask from the soft map instead of
            reading the stored binary, and ``--pp-min-ml`` / ``--pp-min-prob``
            filter components by volume and by peak probability -- the three are
            the sweepable postprocessing knobs, and the latter two need a map.

  compare   Rank several models against each other using the challenge's
            "rank then aggregate" scheme, from their ``per_case.csv`` files.

Predictions and GT must share the native voxel grid (nnU-Net writes native-space
outputs). A predicted mask whose geometry differs from its GT is resampled onto
the GT grid (nearest-neighbor) with a warning.

Usage::

    python scripts/pipeline/evaluate.py evaluate \\
        --pred-dir  .../fold_0/validation \\
        --gt-dir    dataset/nnUNet_raw/Dataset026_ISLES26/labelsTr \\
        --out reports/eval/vanilla_fold0 --name vanilla_fold0

    python scripts/pipeline/evaluate.py compare \\
        vanilla=reports/eval/vanilla_fold0/per_case.csv \\
        resencm=reports/eval/resencm_fold0/per_case.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import re
import sys
from pathlib import Path

import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from isles26.eval import SCORED_METRICS, aggregate, evaluate_case, rank_aggregate
from isles26.postprocessing import postprocess_mask

_SITE_RE = re.compile(r"^sub-(r\d+)s\d+", re.IGNORECASE)


def site_of(case: str) -> str:
    """Site code from a case id: ``sub-r001s001`` -> ``R001``; SOOP -> ``SOOP``."""
    m = _SITE_RE.match(case)
    if m:
        return m.group(1).upper()
    return "SOOP" if "soop" in case.lower() else "unknown"


def size_bin(rec: dict) -> str:
    """Coarse GT lesion-size bucket for breakdowns (illustrative thresholds)."""
    ml = rec["gt_volume_ml"]
    if ml == 0:
        return "empty"
    if ml < 1.0:
        return "small(<1mL)"
    if ml < 10.0:
        return "medium(<10mL)"
    return "large(>=10mL)"


def _load_on_grid(path: Path, reference: sitk.Image, interp: int, case: str) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    if img.GetSize() != reference.GetSize() or img.GetSpacing() != reference.GetSpacing():
        print(f"  [warn] {case}: geometry differs from GT -> resampling onto GT grid", file=sys.stderr)
        img = sitk.Resample(img, reference, sitk.Transform(), interp, 0, img.GetPixelID())
    return img


def _match_cases(pred_dir: Path, gt_dir: Path) -> list[str]:
    def stems(d: Path) -> dict[str, Path]:
        return {p.name[: -len(".nii.gz")]: p for p in d.glob("*.nii.gz")}

    preds, gts = stems(pred_dir), stems(gt_dir)
    shared = sorted(preds.keys() & gts.keys())
    missing = sorted(gts.keys() - preds.keys())
    if missing:
        print(f"[warn] {len(missing)} GT cases have no prediction (skipped): {missing[:5]}...", file=sys.stderr)
    if not shared:
        raise SystemExit(f"No matching cases between {pred_dir} and {gt_dir}")
    return shared


def run_evaluate(args: argparse.Namespace) -> int:
    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    prob_dir = Path(args.prob_dir) if args.prob_dir else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _match_cases(pred_dir, gt_dir)
    rows: list[dict] = []
    for i, case in enumerate(cases, 1):
        gt = sitk.ReadImage(str(gt_dir / f"{case}.nii.gz"))
        pred = _load_on_grid(pred_dir / f"{case}.nii.gz", gt, sitk.sitkNearestNeighbor, case)
        prob = None
        if prob_dir is not None and (prob_dir / f"{case}.nii.gz").exists():
            prob = _load_on_grid(prob_dir / f"{case}.nii.gz", gt, sitk.sitkLinear, case)
        if args.threshold > 0:
            if prob is None:
                raise SystemExit(
                    f"--threshold re-derives the mask from the soft map, but {case} "
                    f"has none under {prob_dir}. Build the prob maps first."
                )
            pred = sitk.Cast(sitk.GreaterEqual(prob, args.threshold), sitk.sitkUInt8)
        pred = postprocess_mask(pred, args.pp_min_ml, args.pp_fill_holes,
                                prob=prob, min_component_prob=args.pp_min_prob,
                                close_mm=args.pp_close_mm)
        # An empty mask means "no lesion" -> flatten the prob map so an empty GT
        # scores PR-AUC 1.0 instead of ~0 on incidental variation. Keyed off
        # emptiness alone, NOT off whether a filter ran: gating it on the filters
        # would hand the unfiltered arm a worse PR-AUC for a reason that has
        # nothing to do with the filters, making sweep arms incomparable.
        if prob is not None and not sitk.GetArrayViewFromImage(pred).any():
            prob = sitk.Multiply(prob, 0.0)
        rec = {"case": case, "site": site_of(case)}
        rec.update(evaluate_case(gt, pred, prob))
        rec["size_bin"] = size_bin(rec)
        rows.append(rec)
        if i % 25 == 0 or i == len(cases):
            # panoptica's result objects are cyclic, so refcounting alone lets RSS
            # climb across a fold -- two runs were OOM-killed in the last 15 cases
            # of 290. Collecting periodically costs nothing next to the scoring.
            gc.collect()
            print(f"  {i}/{len(cases)} scored (last={case})", file=sys.stderr)

    _write_per_case(out_dir / "per_case.csv", rows)
    _write_summary(out_dir / "summary.txt", rows, args.name)
    print(f"\nWrote {out_dir}/per_case.csv and summary.txt ({len(rows)} cases)", file=sys.stderr)
    return 0


def _write_per_case(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _fmt_group(name: str, recs: list[dict]) -> str:
    agg = aggregate(recs)
    parts = [f"{m}={agg[f'{m}_mean']:.4f}" for m in SCORED_METRICS]
    return f"  {name:>16}  n={agg['n_cases']:<5}  " + "  ".join(parts)


def _write_summary(path: Path, rows: list[dict], name: str) -> None:
    lines = [f"Eval summary: {name}", f"cases: {len(rows)}", "", "OVERALL", _fmt_group("all", rows)]
    for key, label in (("site", "BY SITE"), ("size_bin", "BY GT LESION SIZE")):
        lines += ["", label]
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        for g in sorted(groups):
            lines.append(_fmt_group(g, groups[g]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_per_case(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:  # csv is all strings; cast the scored metrics back to float
        for m in SCORED_METRICS:
            r[m] = float(r[m]) if r[m] not in ("", "nan") else float("nan")
    return rows


def run_compare(args: argparse.Namespace) -> int:
    models: dict[str, list[dict]] = {}
    for spec in args.models:
        if "=" not in spec:
            raise SystemExit(f"expected name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        models[name] = _read_per_case(Path(path))

    ranking = rank_aggregate(models)
    print("\nRank-then-aggregate (lower = better):")
    for rank, (name, score) in enumerate(ranking, 1):
        print(f"  {rank}. {name:>16}  mean_rank={score:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate", help="score a prediction dir against GT")
    ev.add_argument("--pred-dir", required=True)
    ev.add_argument("--gt-dir", required=True)
    ev.add_argument("--prob-dir", default=None, help="optional soft prob maps (.nii.gz) for PR-AUC")
    ev.add_argument("--out", required=True)
    ev.add_argument("--name", default="model")
    ev.add_argument("--threshold", type=float, default=0.0,
                    help="re-derive the mask as prob >= THRESHOLD instead of using the "
                         "stored binary mask; requires --prob-dir (0 = use stored mask)")
    ev.add_argument("--pp-min-ml", type=float, default=0.0,
                    help="drop predicted lesions smaller than this many mL before scoring (0 = off)")
    ev.add_argument("--pp-min-prob", type=float, default=0.0,
                    help="drop predicted lesions whose peak probability is below this "
                         "before scoring; requires --prob-dir (0 = off)")
    ev.add_argument("--pp-fill-holes", action="store_true",
                    help="fill holes in the predicted mask before scoring")
    ev.add_argument("--pp-close-mm", type=float, default=0.0,
                    help="binary closing radius in mm applied before components are "
                         "labelled, so fragments of one lesion score as one object (0 = off)")
    ev.set_defaults(func=run_evaluate)

    cmp = sub.add_parser("compare", help="rank models from their per_case.csv files")
    cmp.add_argument("models", nargs="+", help="name=path/to/per_case.csv ...")
    cmp.set_defaults(func=run_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
