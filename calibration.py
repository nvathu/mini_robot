"""calibration.py - fit AFFINE d_true = a * d_pred + b from floor-contact clicks -> scale_shift.json.

Clicks come from camera_height.py annotate (annotations.csv). d_true = forward_m - body_offset.
Same schema as fit_scale.py (selected, scale, shift, usable, d_pred_range, ...), so
point_could_each.py / 3d_generation.py read it unchanged. The old file is kept as scale_shift.old.json.
--source must match the depth that point_could_each.py unprojects (default: resnet student).

  python calibration.py --body-offset 0.07 --source resnet
"""
import argparse
import json
import os
import shutil

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_scale import (bootstrap_ci, collect_contact_samples, cross_validate, fit_affine,
                       load_raw_depths, metrics, spearman, usability)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="dataset/calib/annotations.csv")
    ap.add_argument("--source", choices=["dino", "resnet"], default="resnet")
    ap.add_argument("--out-dir", default="dataset/calib")
    ap.add_argument("--body-offset", type=float, default=0.07,
                    help="m, marker -> floor-contact point; MUST equal the value used in camera_height.py fit")
    ap.add_argument("--patch-u", type=int, default=2)
    ap.add_argument("--patch-v", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.15)
    args = ap.parse_args()

    ann = pd.read_csv(args.annotations, dtype={"img_id": str}).dropna(subset=["u_contact", "v_contact"])
    ann = ann.sort_values("img_id").reset_index(drop=True)

    frames = list(dict.fromkeys(zip(ann["img_id"].astype(str), ann["img_path"])))
    preds = load_raw_depths(frames, args.source)
    S = collect_contact_samples(ann, preds, args.patch_u, args.patch_v, args.body_offset)
    n = len(S)
    if n < 15:
        raise SystemExit(f"Only {n} valid samples; need >= 15.")
    x, y = S["d_pred"].to_numpy(), S["d_true"].to_numpy()

    a, b, inl = fit_affine(x, y, args.thr)
    k = max(2, min(args.folds, n // 4))
    oof_aff, oof_const = cross_validate(x, y, np.arange(n), k, args.thr)   # rows sorted by time
    cv_aff, cv_const = metrics(y, oof_aff), metrics(y, oof_const)
    rho = spearman(x, y)
    a_ci, b_ci = bootstrap_ci(x, y, args.thr)
    reasons = usability(a, a_ci, rho, cv_aff, cv_const)
    usable = not reasons
    lo, hi = float(x.min()), float(x.max())

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "scale_shift.json")
    if os.path.exists(path):
        shutil.copy(path, os.path.join(args.out_dir, "scale_shift.old.json"))
    out = {
        "selected": "affine", "scale": a, "shift": b, "usable": usable, "unusable_reasons": reasons,
        "source": args.source, "n_inliers": int(inl.sum()), "cv": cv_aff, "cv_constant_reference": cv_const,
        "spearman": rho, "bootstrap_ci_95": {"a": a_ci, "b": b_ci},
        "n_samples": int(n), "cv_folds": int(k),
        "d_true_range": [float(y.min()), float(y.max())], "d_pred_range": [lo, hi],
        "d_metric_range_on_pred_range": [a * lo + b, a * hi + b],
        "inlier_thr_m": args.thr, "patch_half_uv": [args.patch_u, args.patch_v],
        "calibration_source": "floor-contact clicks, d_true = forward_m - body_offset",
        "body_offset_m": args.body_offset,
        "warnings": [
            "Validated only at floor-contact points of the robots (one session); not walls/objects.",
            "Outside d_true_range the mapping is extrapolation.",
            "A wrong body_offset shifts b by roughly the offset error.",
        ],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    xs = np.linspace(lo, hi, 50)
    ax1.scatter(x[inl], y[inl], s=16, label="inlier")
    ax1.scatter(x[~inl], y[~inl], s=30, marker="x", c="r", label="outlier")
    ax1.plot(xs, a * xs + b, "b-", label=f"affine a={a:.3f}, b={b:.3f}")
    ax1.set_xlabel(f"d_pred (raw {args.source})"); ax1.set_ylabel("Z_contact (m)")
    ax1.grid(alpha=0.3); ax1.legend(); ax1.set_title(f"spearman={rho:.2f} | usable={usable}")
    ax2.scatter(y, oof_aff - y, s=14, c="b", label=f"affine cv MAE {cv_aff['mae']:.3f}")
    ax2.scatter(y, oof_const - y, s=14, c="gray", label=f"constant cv MAE {cv_const['mae']:.3f}")
    ax2.axhline(0, c="k", lw=0.8); ax2.set_xlabel("Z_contact (m)"); ax2.set_ylabel("out-of-fold error (m)")
    ax2.grid(alpha=0.3); ax2.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "scale_shift_check.png"), dpi=140)

    print(f"n={n} | source={args.source} | affine a={a:.4f} b={b:.4f} | inliers {int(inl.sum())}/{n}")
    print(f"bootstrap 95% CI: a [{a_ci[0]:.3f}, {a_ci[1]:.3f}], b [{b_ci[0]:.3f}, {b_ci[1]:.3f}] | spearman={rho:.3f}")
    print(f"held-out: affine MAE={cv_aff['mae']:.3f} m, rel median={cv_aff['median_rel_err'] * 100:.1f}%, "
          f"bias={cv_aff['bias']:+.3f} | constant MAE={cv_const['mae']:.3f} m")
    print(f"d_true=[{y.min():.2f}, {y.max():.2f}] m | d_pred=[{lo:.3f}, {hi:.3f}] -> metric "
          f"[{a * lo + b:.2f}, {a * hi + b:.2f}] m")
    print("USABLE: " + ("yes" if usable else "NO -> " + "; ".join(reasons)))
    print(f"Wrote {path} (old file -> scale_shift.old.json)")


if __name__ == "__main__":
    main()