"""Paired significance tests for the connected-component volume threshold.

Answers whether the tau = 0 -> tau = 0.05 mL change in Table 1 is statistically
significant, per metric and per backbone. Emits markdown tables.

    python docs/paper/stats_postprocessing.py

Both arms score the same weights over the same 1450 cross-validation cases, so
every case contributes a matched pair and every test is paired.

Three inferences are reported per metric because they answer different
questions. The paired t-test is on the MEAN difference, which is what Table 1
tabulates. The bootstrap CI is also on the mean but assumes no distribution,
which matters where the differences are skewed enough that n = 1450 does not
buy normality of the sampling mean. Wilcoxon signed-rank is on rank balance and
is insensitive to a few large outliers. Agreement across the three is the claim;
disagreement localises what is driving the change.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, ttest_rel, wilcoxon

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "reports" / "eval"
TAU = {"0": "Dataset026_ISLES26_pan2", "0.05": "Dataset026_ISLES26_pp0.05_pan2"}
MODELS = {"ResEnc M": "resencm", "Default": "vanilla"}

METRICS = ["dice", "abs_volume_diff_ml", "pr_auc", "detection_f1", "abs_count_diff"]
LABEL = {"dice": "Dice", "abs_volume_diff_ml": "Abs. volume diff (mL)",
         "pr_auc": "PR-AUC", "detection_f1": "Detection F1",
         "abs_count_diff": "Abs. lesion-count diff"}
HIGHER_BETTER = {"dice", "pr_auc", "detection_f1"}
# Shift observed between two runs of the identical config differing only in
# random seed (scripts/slurm/seed_noise.slurm, fold 0). Used as a practical
# floor: a change smaller than this is not distinguishable from a reseed.
SEED_FLOOR = {"dice": 0.0018, "abs_volume_diff_ml": 0.0120, "pr_auc": 0.0192,
              "detection_f1": 0.0109, "abs_count_diff": 0.0379}
N_BOOT = 1000


def _read(model: str, tau: str) -> dict[str, dict[str, float]]:
    path = EVAL / TAU[tau] / f"{MODELS[model]}_cv" / "per_case.csv"
    rows = csv.DictReader(path.open(encoding="utf-8"))
    return {r["case"]: {m: float(r[m]) for m in METRICS} for r in rows}


def _holm(pvals: list[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values."""
    adj, running = np.empty(len(pvals)), 0.0
    for i, idx in enumerate(np.argsort(pvals)):
        running = max(running, (len(pvals) - i) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def compare(model: str, rng: np.random.Generator) -> list[dict]:
    a, b = _read(model, "0"), _read(model, "0.05")
    cases = sorted(set(a) & set(b))
    out, p_w, p_t = [], [], []
    for m in METRICS:
        x = np.array([a[c][m] for c in cases])
        y = np.array([b[c][m] for c in cases])
        d = y - x                       # tau=0.05 minus tau=0
        nz = d[d != 0]
        p = wilcoxon(y, x, zero_method="wilcox").pvalue if nz.size else 1.0
        pt = ttest_rel(y, x).pvalue if nz.size else 1.0
        # Matched-pairs rank-biserial correlation: signed-rank sums over the
        # non-tied pairs. rankdata averages ties, which argsort would not.
        if nz.size:
            r = rankdata(np.abs(nz))
            pos, neg = r[nz > 0].sum(), r[nz < 0].sum()
            rb = (pos - neg) / (pos + neg)
        else:
            rb = 0.0
        boot = np.array([rng.choice(d, d.size, replace=True).mean() for _ in range(N_BOOT)])
        better = int((d > 0).sum() if m in HIGHER_BETTER else (d < 0).sum())
        worse = int((d < 0).sum() if m in HIGHER_BETTER else (d > 0).sum())
        out.append({"metric": m, "n_changed": int(nz.size), "mean": float(d.mean()),
                    "ci": tuple(np.percentile(boot, [2.5, 97.5])), "rb": float(rb),
                    "better": better, "worse": worse})
        p_w.append(p)
        p_t.append(pt)
    # Holm within each test family: 5 metrics, one family per test.
    for row, pw, pt in zip(out, _holm(p_w), _holm(p_t)):
        row["p_wilcoxon"] = float(pw)
        row["p_ttest"] = float(pt)
    return out


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _verdict(row: dict) -> str:
    m, mean = row["metric"], row["mean"]
    # Both tests must agree before the difference is called real.
    if max(row["p_ttest"], row["p_wilcoxon"]) >= 0.05:
        return "no difference" if min(row["p_ttest"], row["p_wilcoxon"]) >= 0.05 else "tests disagree"
    improves = mean > 0 if m in HIGHER_BETTER else mean < 0
    tag = "improves" if improves else "worsens"
    return tag if abs(mean) > SEED_FLOOR[m] else f"{tag}, below seed floor"


def main() -> None:
    rng = np.random.default_rng(0)
    for model in MODELS:
        print(f"\n### {model}: tau = 0 vs tau = 0.05 mL\n")
        print("| Metric | Mean diff | 95% CI (bootstrap) | Changed | Better/worse | "
              "Rank-biserial | p (t-test) | p (Wilcoxon) | Reading |")
        print("|---|---|---|---|---|---|---|---|---|")
        for row in compare(model, rng):
            print(f"| {LABEL[row['metric']]} | {row['mean']:+.4f} | "
                  f"[{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}] | {row['n_changed']} | "
                  f"{row['better']}/{row['worse']} | {row['rb']:+.3f} | "
                  f"{_fmt_p(row['p_ttest'])} | {_fmt_p(row['p_wilcoxon'])} | "
                  f"{_verdict(row)} |")


if __name__ == "__main__":
    main()
