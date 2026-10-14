# fit_scale_shift.py
# OFFLINE calibration: fit d_true = a * d_pred + b on manually validated frames (labels.csv).
# Compares scale-only vs affine with contiguous-block cross-validation, saves scale_shift.json.
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS_CSV = "dataset/valid_frames/labels.csv"
WEIGHTS_PATH = "weights/best_model.pth"
OUT_DIR = "dataset/calib"
NET_SIZE = 180
ROBOTS = ["S1", "S2"]


def predict_raw_depths(img_paths):
    import torch
    from model import ResNetDepth
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetDepth().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    out = {}
    with torch.no_grad():
        for p in img_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            rgb = cv2.cvtColor(cv2.resize(img, (NET_SIZE, NET_SIZE)), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
            out[p] = (model(t).squeeze().cpu().numpy(), img.shape[:2])
    return out


def collect_samples(labels, preds, patch_u, patch_v):
    rows = []
    for _, row in labels.iterrows():
        p = row["img_path"]
        if p not in preds:
            continue
        depth, (h0, w0) = preds[p]
        for rb in ROBOTS:
            if not int(row[f"{rb}_valid"]):
                continue
            ui = int(round(row[f"{rb}_u"] * NET_SIZE / w0))
            vi = int(round(row[f"{rb}_v"] * NET_SIZE / h0))
            patch = depth[max(vi - patch_v, 0):min(vi + patch_v + 1, NET_SIZE),
                          max(ui - patch_u, 0):min(ui + patch_u + 1, NET_SIZE)]
            if patch.size == 0:
                continue
            med = float(np.median(patch))
            q25, q75 = np.percentile(patch, [25, 75])
            if not np.isfinite(med) or med <= 1e-3:
                continue
            rows.append({"img_id": str(row["img_id"]), "robot": rb, "csv_idx": int(row["csv_idx"]),
                         "d_true": float(row[f"{rb}_forward_m"]), "d_pred": med,
                         "patch_iqr": float(q75 - q25)})
    return pd.DataFrame(rows)


def msac_cost(res, thr):
    return np.minimum(res ** 2, thr ** 2).sum(axis=-1)


def lstsq_affine(x, y):
    A = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def fit_scale_only(x, y, thr):
    cand = y / x
    res = np.abs(cand[:, None] * x[None, :] - y[None, :])
    a = cand[np.argmin(msac_cost(res, thr))]
    inl = np.abs(a * x - y) < thr
    if inl.any():
        a = (x[inl] @ y[inl]) / (x[inl] @ x[inl])
    inl = np.abs(a * x - y) < thr
    return float(a), 0.0, inl


def fit_affine(x, y, thr, iters=3000, seed=0):
    n = len(x)
    if n < 3:
        a, b = lstsq_affine(x, y)
        return a, b, np.abs(a * x + b - y) < thr
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, n, iters), rng.integers(0, n, iters)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 0.1 * np.ptp(x)  # avoid ill-conditioned pairs
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
    inl = np.abs(a0 * x + b0 - y) < thr
    return a0, b0, inl


def metrics(y, p):
    e = p - y
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "medae": float(np.median(np.abs(e))), "median_rel_err": float(np.median(np.abs(e) / y))}


def cross_validate(x, y, order, k, thr):
    folds = np.array_split(order, k)
    oof = {"scale_only": np.full(len(x), np.nan), "affine": np.full(len(x), np.nan)}
    for f in folds:
        tr = np.setdiff1d(order, f)
        a, b, _ = fit_scale_only(x[tr], y[tr], thr)
        oof["scale_only"][f] = a * x[f] + b
        a, b, _ = fit_affine(x[tr], y[tr], thr)
        oof["affine"][f] = a * x[f] + b
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-u", type=int, default=3, help="half-width of depth patch (net pixels)")
    ap.add_argument("--patch-v", type=int, default=6, help="half-height of depth patch (net pixels)")
    ap.add_argument("--inlier-thr", type=float, default=0.15, help="RANSAC inlier threshold (m)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    labels = pd.read_csv(LABELS_CSV, dtype={"img_id": str})
    labels = labels[(labels["S1_valid"] == 1) | (labels["S2_valid"] == 1)].reset_index(drop=True)
    print(f"Valid frames: {len(labels)}")

    preds = predict_raw_depths(labels["img_path"].unique())
    S = collect_samples(labels, preds, args.patch_u, args.patch_v)
    n = len(S)
    if n < 6:
        raise SystemExit(f"Only {n} usable samples - not enough to fit.")
    x, y = S["d_pred"].to_numpy(), S["d_true"].to_numpy()
    print(f"Samples: {n} (S1={int((S.robot == 'S1').sum())}, S2={int((S.robot == 'S2').sum())})")
    print(f"d_true range: {y.min():.2f} - {y.max():.2f} m | d_pred range: {x.min():.3f} - {x.max():.3f}")
    if np.ptp(y) < 0.5:
        print("WARNING: d_true spans <0.5 m -> shift b is ill-conditioned, extrapolation to the room is unreliable")

    thr = args.inlier_thr
    a_s, b_s, inl_s = fit_scale_only(x, y, thr)
    a_a, b_a, inl_a = fit_affine(x, y, thr)

    k = max(2, min(args.folds, n // 4))
    order = np.argsort(S["csv_idx"].to_numpy(), kind="stable")
    oof = cross_validate(x, y, order, k, thr)
    cv_s, cv_a = metrics(y, oof["scale_only"]), metrics(y, oof["affine"])

    use_affine = a_a > 0 and cv_a["mae"] < 0.95 * cv_s["mae"]
    selected = "affine" if use_affine else "scale_only"
    a_sel, b_sel = (a_a, b_a) if use_affine else (a_s, b_s)
    inl_sel = inl_a if use_affine else inl_s

    print(f"\n{'model':<11}{'a':>8}{'b':>9}{'inliers':>9}{'cvMAE':>8}{'cvRMSE':>8}{'cvMedAE':>9}{'cvRelErr':>9}")
    for name, a, b, inl, m in (("scale_only", a_s, b_s, inl_s, cv_s), ("affine", a_a, b_a, inl_a, cv_a)):
        print(f"{name:<11}{a:8.4f}{b:9.4f}{int(inl.sum()):9d}{m['mae']:8.3f}{m['rmse']:8.3f}"
              f"{m['medae']:9.3f}{m['median_rel_err']:9.3f}")
    print(f"\nSelected: {selected} (a={a_sel:.4f}, b={b_sel:.4f}); affine needs >5% lower CV MAE to win")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"selected": selected, "scale": a_sel, "shift": b_sel,
           "scale_only": {"a": a_s, "b": 0.0, "n_inliers": int(inl_s.sum()), "cv": cv_s},
           "affine": {"a": a_a, "b": b_a, "n_inliers": int(inl_a.sum()), "cv": cv_a},
           "n_samples": int(n), "n_frames": int(len(labels)), "cv_folds": int(k),
           "d_true_range": [float(y.min()), float(y.max())],
           "d_pred_range": [float(x.min()), float(x.max())],
           "inlier_thr_m": thr, "patch_half_uv": [args.patch_u, args.patch_v]}
    with open(os.path.join(OUT_DIR, "scale_shift.json"), "w") as f:
        json.dump(out, f, indent=2)

    S["inlier_selected"] = inl_sel
    S["oof_scale_only"], S["oof_affine"] = oof["scale_only"], oof["affine"]
    S.to_csv(os.path.join(OUT_DIR, "scale_shift_samples.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    xs = np.linspace(0, x.max() * 1.1, 50)
    for rb, col in (("S1", "tab:green"), ("S2", "tab:red")):
        m = (S["robot"] == rb).to_numpy()
        ax1.scatter(x[m & inl_sel], y[m & inl_sel], c=col, s=18, label=f"{rb} inlier")
        ax1.scatter(x[m & ~inl_sel], y[m & ~inl_sel], c=col, marker="x", s=30, label=f"{rb} outlier")
    ax1.plot(xs, a_s * xs, "k--", label=f"scale-only a={a_s:.3f}")
    ax1.plot(xs, a_a * xs + b_a, "b-", label=f"affine a={a_a:.3f}, b={b_a:.3f}")
    ax1.set_xlabel("d_pred (raw net output)"); ax1.set_ylabel("d_true (MoCap, m)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3); ax1.set_title("Scale-shift fit")
    ax2.scatter(y, oof["scale_only"] - y, s=14, c="k", label=f"scale-only (cv MAE {cv_s['mae']:.3f})")
    ax2.scatter(y, oof["affine"] - y, s=14, c="b", label=f"affine (cv MAE {cv_a['mae']:.3f})")
    ax2.axhline(0, color="gray")
    ax2.set_xlabel("d_true (m)"); ax2.set_ylabel("out-of-fold error (m)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_title("Held-out residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scale_shift_check.png"), dpi=140)
    print(f"Saved: {OUT_DIR}/scale_shift.json, scale_shift_samples.csv, scale_shift_check.png")


if __name__ == "__main__":
    main()