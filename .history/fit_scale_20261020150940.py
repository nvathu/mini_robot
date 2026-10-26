# fit_scale.py
# OFFLINE calibration: fit d_true = a * d_pred + b (AFFINE ONLY) on manually validated frames.
# Robust fit (MSAC + inlier refit), contiguous-block CV, bootstrap CI, and a usability flag.
# MoCap is used here only to build the calibration; never at inference time.
#
#   python fit_scale.py                  # fit on DINOv3 depth (same source as point_could_each.py)
#   python fit_scale.py --source resnet  # fit on ResNet student output instead
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SESSION = "2025_0813_1543-S3"
LABELS_CSV = "dataset/valid_frames/labels.csv"
DEPTH_ROOT = Path("dataset/depth_dino") / SESSION
WEIGHTS_PATH = "weights/best_model.pth"
OUT_DIR = "dataset/calib"
NET_SIZE = 180          # reference size for --patch-u / --patch-v
ROBOTS = ["S1", "S2"]


# --------------------------------------------------
# Raw depth source
# --------------------------------------------------

def load_raw_depths(frames, source):
    """frames: list of (img_id, img_path). Returns {img_id: (depth HxW, (h0, w0) of RGB)}."""
    model = device = torch = None
    if source == "resnet":
        import torch
        from model import ResNetDepth
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ResNetDepth().to(device)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model.eval()

    out = {}
    for img_id, path in frames:
        img = cv2.imread(path)
        if img is None:
            continue
        shape = img.shape[:2]
        if source == "dino":
            dp = DEPTH_ROOT / f"image_{img_id}_depth.npy"
            if not dp.exists():
                continue
            depth = np.squeeze(np.load(dp)).astype(np.float32)
            if depth.ndim != 2:
                continue
        else:
            rgb = cv2.cvtColor(cv2.resize(img, (NET_SIZE, NET_SIZE)), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                depth = model(t).squeeze().cpu().numpy()
        out[img_id] = (depth, shape)
    return out


def collect_samples(labels, preds, patch_u, patch_v):
    rows = []
    for _, row in labels.iterrows():
        key = str(row["img_id"])
        if key not in preds:
            continue
        depth, (h0, w0) = preds[key]
        Hd, Wd = depth.shape
        pu = max(1, int(round(patch_u * Wd / NET_SIZE)))
        pv = max(1, int(round(patch_v * Hd / NET_SIZE)))
        for rb in ROBOTS:
            if not int(row[f"{rb}_valid"]):
                continue
            ui = int(round(row[f"{rb}_u"] * Wd / w0))
            vi = int(round(row[f"{rb}_v"] * Hd / h0))
            patch = depth[max(vi - pv, 0):min(vi + pv + 1, Hd), max(ui - pu, 0):min(ui + pu + 1, Wd)]
            if patch.size == 0:
                continue
            med = float(np.median(patch))
            q25, q75 = np.percentile(patch, [25, 75])
            if not np.isfinite(med) or med <= 1e-3:
                continue
            rows.append({"img_id": key, "robot": rb, "csv_idx": int(row["csv_idx"]),
                         "d_true": float(row[f"{rb}_forward_m"]), "d_pred": med,
                         "patch_iqr": float(q75 - q25)})
    return pd.DataFrame(rows)


# --------------------------------------------------
# Robust affine fit
# --------------------------------------------------

def msac_cost(res, thr):
    return np.minimum(res ** 2, thr ** 2).sum(axis=-1)


def lstsq_affine(x, y):
    A = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def fit_affine(x, y, thr, iters=3000, seed=0):
    """MSAC over 2-point hypotheses (a > 0 only), then least squares on the inliers."""
    n = len(x)
    if n < 3:
        a, b = lstsq_affine(x, y)
        return a, b, np.abs(a * x + b - y) < thr
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, n, iters), rng.integers(0, n, iters)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 0.1 * np.ptp(x)
    i, j, dx = i[ok], j[ok], dx[ok]
    if len(i) == 0:
        a, b = lstsq_affine(x, y)
        return a, b, np.abs(a * x + b - y) < thr
    a = (y[j] - y[i]) / dx
    b = y[i] - a * x[i]
    good = a > 0
    a, b = a[good], b[good]
    if len(a) == 0:
        a0, b0 = lstsq_affine(x, y)
        return a0, b0, np.abs(a0 * x + b0 - y) < thr
    res = np.abs(a[:, None] * x[None, :] + b[:, None] - y[None, :])
    k = np.argmin(msac_cost(res, thr))
    a0, b0 = float(a[k]), float(b[k])
    inl = np.abs(a0 * x + b0 - y) < thr
    if inl.sum() >= 3:
        a0, b0 = lstsq_affine(x[inl], y[inl])
    return a0, b0, np.abs(a0 * x + b0 - y) < thr


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def metrics(y, p):
    e = p - y
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "medae": float(np.median(np.abs(e))), "median_rel_err": float(np.median(np.abs(e) / y)),
            "bias": float(np.mean(e))}


def cross_validate(x, y, order, k, thr):
    """Contiguous blocks (by csv_idx) so correlated neighbouring frames never straddle train/test."""
    oof_aff, oof_const = np.full(len(x), np.nan), np.full(len(x), np.nan)
    for f in np.array_split(order, k):
        tr = np.setdiff1d(order, f)
        a, b, _ = fit_affine(x[tr], y[tr], thr)
        oof_aff[f] = a * x[f] + b
        oof_const[f] = np.median(y[tr])       # reference: ignores the network entirely
    return oof_aff, oof_const


def bootstrap_ci(x, y, thr, n_boot=300, seed=1):
    rng = np.random.default_rng(seed)
    ab = []
    for s in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        a, b, _ = fit_affine(x[idx], y[idx], thr, iters=500, seed=s)
        ab.append((a, b))
    ab = np.asarray(ab)
    return np.percentile(ab[:, 0], [2.5, 97.5]).tolist(), np.percentile(ab[:, 1], [2.5, 97.5]).tolist()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dino", "resnet"], default="dino",
                    help="raw depth to calibrate; must match what point_could_each.py unprojects")
    ap.add_argument("--patch-u", type=int, default=3, help="half-width of depth patch (180-px units)")
    ap.add_argument("--patch-v", type=int, default=6, help="half-height of depth patch (180-px units)")
    ap.add_argument("--inlier-thr", type=float, default=0.15, help="RANSAC inlier threshold (m)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    labels = pd.read_csv(LABELS_CSV, dtype={"img_id": str})
    labels = labels[(labels["S1_valid"] == 1) | (labels["S2_valid"] == 1)].reset_index(drop=True)
    print(f"Valid frames: {len(labels)} | source: {args.source}")

    frames = list(zip(labels["img_id"].astype(str), labels["img_path"]))
    preds = load_raw_depths(frames, args.source)
    S = collect_samples(labels, preds, args.patch_u, args.patch_v)
    n = len(S)
    if n < 6:
        raise SystemExit(f"Only {n} usable samples - not enough to fit.")
    x, y = S["d_pred"].to_numpy(), S["d_true"].to_numpy()
    print(f"Samples: {n} (S1={int((S.robot == 'S1').sum())}, S2={int((S.robot == 'S2').sum())})")
    print(f"d_true range: {y.min():.2f} - {y.max():.2f} m | d_pred range: {x.min():.3f} - {x.max():.3f}")

    thr = args.inlier_thr
    a, b, inl = fit_affine(x, y, thr)
    k = max(2, min(args.folds, n // 4))
    order = np.argsort(S["csv_idx"].to_numpy(), kind="stable")
    oof_aff, oof_const = cross_validate(x, y, order, k, thr)
    cv_aff, cv_const = metrics(y, oof_aff), metrics(y, oof_const)
    rho = spearman(x, y)
    a_ci, b_ci = bootstrap_ci(x, y, thr)

    lo, hi = float(x.min()), float(x.max())
    metric_span = [a * lo + b, a * hi + b]

    # usable = network carries distance information AND affine beats the constant reference
    reasons = []
    if a <= 0:
        reasons.append("a <= 0")
    if rho < 0.3:
        reasons.append(f"spearman(d_pred, d_true) = {rho:.2f} < 0.3")
    if cv_aff["mae"] > 0.9 * cv_const["mae"]:
        reasons.append("CV MAE not >10% better than predicting a constant")
    if a_ci[0] <= 0:
        reasons.append("bootstrap CI of a includes <= 0")
    usable = len(reasons) == 0

    print(f"\naffine: a={a:.4f} b={b:.4f} | inliers {int(inl.sum())}/{n}")
    print(f"bootstrap 95% CI: a in [{a_ci[0]:.3f}, {a_ci[1]:.3f}], b in [{b_ci[0]:.3f}, {b_ci[1]:.3f}]")
    print(f"spearman(d_pred, d_true) = {rho:.3f}")
    print(f"{'model':<16}{'cvMAE':>8}{'cvRMSE':>8}{'cvMedAE':>9}{'cvRelErr':>9}{'bias':>8}")
    for name, m in (("affine", cv_aff), ("const (ref)", cv_const)):
        print(f"{name:<16}{m['mae']:8.3f}{m['rmse']:8.3f}{m['medae']:9.3f}{m['median_rel_err']:9.3f}{m['bias']:8.3f}")
    print(f"Metric depth produced over the whole raw range: {metric_span[0]:.2f} - {metric_span[1]:.2f} m "
          f"(span {metric_span[1] - metric_span[0]:.2f} m)")
    if usable:
        print("USABLE: yes")
    else:
        print("USABLE: NO -> " + "; ".join(reasons))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"selected": "affine", "scale": a, "shift": b, "usable": usable, "unusable_reasons": reasons,
           "source": args.source, "n_inliers": int(inl.sum()), "cv": cv_aff, "cv_constant_reference": cv_const,
           "spearman": rho, "bootstrap_ci_95": {"a": a_ci, "b": b_ci},
           "n_samples": int(n), "n_frames": int(len(labels)), "cv_folds": int(k),
           "d_true_range": [float(y.min()), float(y.max())], "d_pred_range": [lo, hi],
           "d_metric_range_on_pred_range": metric_span,
           "inlier_thr_m": thr, "patch_half_uv": [args.patch_u, args.patch_v]}
    with open(os.path.join(OUT_DIR, "scale_shift.json"), "w") as f:
        json.dump(out, f, indent=2)

    S["inlier"] = inl
    S["oof_affine"] = oof_aff
    S.to_csv(os.path.join(OUT_DIR, "scale_shift_samples.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    xs = np.linspace(x.min() - 0.05 * np.ptp(x), x.max() + 0.05 * np.ptp(x), 50)
    for rb, col in (("S1", "tab:green"), ("S2", "tab:red")):
        m = (S["robot"] == rb).to_numpy()
        ax1.scatter(x[m & inl], y[m & inl], c=col, s=18, label=f"{rb} inlier")
        ax1.scatter(x[m & ~inl], y[m & ~inl], c=col, marker="x", s=30, label=f"{rb} outlier")
    ax1.plot(xs, a * xs + b, "b-", label=f"affine a={a:.3f}, b={b:.3f}")
    ax1.axhline(np.median(y), color="gray", ls="--", label="constant reference")
    ax1.set_xlabel(f"d_pred (raw {args.source} output)"); ax1.set_ylabel("d_true (MoCap, m)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_title(f"Affine fit | spearman={rho:.2f} | usable={usable}")
    ax2.scatter(y, oof_aff - y, s=14, c="b", label=f"affine (cv MAE {cv_aff['mae']:.3f})")
    ax2.scatter(y, oof_const - y, s=14, c="gray", label=f"constant (cv MAE {cv_const['mae']:.3f})")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("d_true (m)"); ax2.set_ylabel("out-of-fold error (m)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_title("Held-out residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scale_shift_check.png"), dpi=140)
    print(f"Saved: {OUT_DIR}/scale_shift.json, scale_shift_samples.csv, scale_shift_check.png")


if __name__ == "__main__":
    main()