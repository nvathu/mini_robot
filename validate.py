"""validate_model.py - OFFLINE accuracy checks of the metric depth / point clouds (uses MoCap + floor geometry).

T1 contact : depth at floor-contact clicks vs MoCap distance (forward_m - body_offset), per distance bin.
             Use a click set NOT used in calibration, or --cv (blocked cross-validation on the same clicks).
T2 floor   : dense depth vs geometric floor model Z = fy * h / (v - cy)  (needs camera height, pitch = 0).
T3 consist : multi-view cloud-to-cloud distance above the floor (needs scipy), with a shifted negative control.

  python validate_model.py --tests all
  python validate_model.py --tests contact --annotations dataset/calib/annotations_test.csv
  python validate_model.py --tests contact --cv
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from depth_source import DepthSource
from fit_scale import (collect_contact_samples, cross_validate, fit_affine, load_raw_depths,
                       metrics)

SESSION = "2025_0813_1543-S3"
LABELS_CSV = "dataset/valid_frames/labels.csv"
INTRINSICS_PATH = "dataset/calib/intrinsics.json"
SCALE_SHIFT_PATH = "dataset/calib/scale_shift.json"
CAM_HEIGHT_PATH = "dataset/calib/camera_height.json"
CLOUD_DIR = Path("dataset/pointclouds") / SESSION
OUT_DIR = "dataset/calib"


def bin_table(true, err, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (true >= lo) & (true < hi)
        if m.sum() == 0:
            continue
        e = err[m]
        rows.append({"range_m": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                     "median_err_m": float(np.median(e)), "mae_m": float(np.mean(np.abs(e))),
                     "median_rel_%": float(100 * np.median(np.abs(e) / true[m]))})
    return rows


def print_table(rows):
    print(f"  {'range (m)':<10}{'n':>5}{'median err':>12}{'MAE':>8}{'rel %':>8}")
    for r in rows:
        print(f"  {r['range_m']:<10}{r['n']:>5}{r['median_err_m']:>+12.3f}{r['mae_m']:>8.3f}{r['median_rel_%']:>8.1f}")


def load_intrinsics(w, h):
    with open(INTRINSICS_PATH) as f:
        K = json.load(f)
    sx, sy = w / float(K.get("width", 640)), h / float(K.get("height", 480))
    return K["fx"] * sx, K["fy"] * sy, K["cx"] * sx, K["cy"] * sy


# --------------------------------------------------
# T1: depth at floor-contact points vs MoCap
# --------------------------------------------------

def test_contact(args, calib):
    print("\n[T1] floor-contact depth vs MoCap")
    ann = pd.read_csv(args.annotations, dtype={"img_id": str}).dropna(subset=["u_contact", "v_contact"])
    ann = ann.sort_values("img_id").reset_index(drop=True)
    frames = list(dict.fromkeys(zip(ann["img_id"].astype(str), ann["img_path"])))
    preds = load_raw_depths(frames, args.source)
    S = collect_contact_samples(ann, preds, 2, 4, args.body_offset)
    n = len(S)
    if n < 8:
        print(f"  only {n} usable samples -> skipped")
        return None
    x, y = S["d_pred"].to_numpy(), S["d_true"].to_numpy()
    if args.cv:
        k = max(2, min(5, n // 4))
        pred, _ = cross_validate(x, y, np.arange(n), k, 0.15)
        mode = f"blocked {k}-fold CV (same clicks as calibration)"
    else:
        pred = calib["scale"] * x + calib["shift"]
        mode = "applying scale_shift.json (clicks must be disjoint from calibration)"
    err = pred - y
    m = metrics(y, pred)
    base = metrics(y, np.full_like(y, np.median(y)))
    print(f"  mode: {mode}\n  n={n}  d_true {y.min():.2f}-{y.max():.2f} m")
    print(f"  MAE={m['mae']:.3f} m  RMSE={m['rmse']:.3f}  median rel err={100 * m['median_rel_err']:.1f}%  "
          f"bias={m['bias']:+.3f}   | constant baseline MAE={base['mae']:.3f} m")
    rows = bin_table(y, err, np.array([0.3, 0.6, 1.0, 1.4, 2.1]))
    print_table(rows)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y, pred, s=16)
    lim = [0, max(y.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("MoCap Z_contact (m)"); ax.set_ylabel("predicted metric depth (m)"); ax.grid(alpha=0.3)
    ax.set_title("T1 contact depth")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "validation_contact.png"), dpi=140)
    return {"mode": mode, "n": int(n), "metrics": m, "constant_baseline": base, "bins": rows}


# --------------------------------------------------
# T2: dense depth vs geometric floor model
# --------------------------------------------------

def test_floor(args, calib):
    print("\n[T2] dense depth vs floor-plane model")
    h_c = args.cam_height
    if h_c is None:
        with open(CAM_HEIGHT_PATH) as f:
            h_c = float(json.load(f)["h_c_m"])
    labels = pd.read_csv(LABELS_CSV, dtype={"img_id": str})
    labels = labels[(labels["S1_valid"] == 1) | (labels["S2_valid"] == 1)].iloc[::max(1, args.every)]
    a, b = calib["scale"], calib["shift"]
    src = DepthSource(args.source)
    z_all, d_all = [], []
    for _, r in labels.iterrows():
        img = cv2.imread(str(r["img_path"]))
        if img is None:
            continue
        raw = src.depth(r["img_id"], img)
        if raw is None:
            continue
        H, W = img.shape[:2]
        fx, fy, cx, cy = load_intrinsics(W, H)
        depth = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
        metric = a * depth + b
        v = np.arange(int(cy) + 5, H)
        z_floor = fy * h_c / (v - cy)
        keep = (z_floor >= args.zmin) & (z_floor <= args.zmax)
        v, z_floor = v[keep], z_floor[keep]
        if len(v) == 0:
            continue
        u0, u1 = int(cx - args.band), int(cx + args.band)
        d_row = np.median(metric[v][:, u0:u1], axis=1)      # median over the band: robust to robots
        z_all.append(z_floor); d_all.append(d_row)
    if not z_all:
        print("  no frames usable -> skipped")
        return None
    z, d = np.concatenate(z_all), np.concatenate(d_all)
    err = d - z
    print(f"  h_c={h_c:.3f} m, frames={len(z_all)}, rows={len(z)}, floor Z {z.min():.2f}-{z.max():.2f} m")
    print(f"  median err={np.median(err):+.3f} m  MAE={np.mean(np.abs(err)):.3f} m  "
          f"median rel={100 * np.median(np.abs(err) / z):.1f}%")
    rows = bin_table(z, err, np.array([0.3, 0.6, 1.0, 1.4, 2.1]))
    print_table(rows)
    print("  note: only as independent as h_c; Z_floor scales linearly with h_c (see its CI). "
          "Rows containing robots/walls inflate the error.")
    return {"h_c_m": h_c, "n_rows": int(len(z)), "median_err_m": float(np.median(err)),
            "mae_m": float(np.mean(np.abs(err))), "bins": rows}


# --------------------------------------------------
# T3: multi-view consistency
# --------------------------------------------------

def test_consistency(args, calib):
    print("\n[T3] multi-view cloud-to-cloud consistency (height > %.2f m)" % args.min_height)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("  scipy missing (pip install scipy) -> skipped")
        return None
    rng = np.random.default_rng(0)
    clouds = []
    for p in sorted(CLOUD_DIR.glob("image_*.npz")):
        with np.load(p, allow_pickle=False) as d:
            pts = d["points_world"]
            if len(pts) == 0:
                continue
            pts = pts[pts[:, 1] > args.min_height]
            if len(pts) < 500:
                continue
            if len(pts) > args.max_points:
                pts = pts[rng.choice(len(pts), args.max_points, replace=False)]
            clouds.append((pts, d["cam_pos_world"], float(d["yaw_deg"])))
    n = len(clouds)
    if n < 2:
        print("  < 2 clouds -> skipped")
        return None
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if np.hypot(*(clouds[i][1][[0, 2]] - clouds[j][1][[0, 2]])) < args.max_cam_dist
             and abs((clouds[i][2] - clouds[j][2] + 180) % 360 - 180) < args.max_yaw_diff]
    if len(pairs) > args.pairs:
        pairs = [pairs[k] for k in rng.choice(len(pairs), args.pairs, replace=False)]
    med, med_ctrl, ov = [], [], []
    shift = np.array([0.10, 0.0, 0.10])
    for i, j in pairs:
        A, B = clouds[i][0], clouds[j][0]
        tree = cKDTree(B)
        d, _ = tree.query(A, distance_upper_bound=args.radius)
        m = np.isfinite(d)
        if m.mean() < 0.1:
            continue
        d2, _ = cKDTree(B + shift).query(A, distance_upper_bound=args.radius)
        m2 = np.isfinite(d2)
        if m2.sum() == 0:
            continue
        med.append(float(np.median(d[m]))); ov.append(float(m.mean())); med_ctrl.append(float(np.median(d2[m2])))
    if not med:
        print("  no overlapping pairs found -> skipped (try larger --max-cam-dist / --max-yaw-diff)")
        return None
    print(f"  frames={n}, candidate pairs={len(pairs)}, used={len(med)}, overlap fraction (median)={np.median(ov):.2f}")
    print(f"  median NN distance = {np.median(med) * 100:.1f} cm (p90 {np.percentile(med, 90) * 100:.1f} cm)")
    print(f"  negative control (B shifted 14 cm) = {np.median(med_ctrl) * 100:.1f} cm "
          "-> the metric is meaningful only if this is clearly larger")
    return {"pairs_used": len(med), "median_nn_cm": float(np.median(med) * 100),
            "p90_nn_cm": float(np.percentile(med, 90) * 100),
            "control_nn_cm": float(np.median(med_ctrl) * 100), "overlap_median": float(np.median(ov))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="all", help="all | contact,floor,consist (comma list)")
    ap.add_argument("--source", choices=["dino", "resnet"], default="resnet")
    ap.add_argument("--annotations", default="dataset/calib/annotations.csv")
    ap.add_argument("--cv", action="store_true", help="T1: blocked CV instead of applying scale_shift.json")
    ap.add_argument("--body-offset", type=float, default=0.07)
    ap.add_argument("--cam-height", type=float, default=None)
    ap.add_argument("--every", type=int, default=1, help="T2: take every N-th valid frame")
    ap.add_argument("--zmin", type=float, default=0.4)
    ap.add_argument("--zmax", type=float, default=2.0)
    ap.add_argument("--band", type=int, default=100, help="T2: half-width (px) of the central column band")
    ap.add_argument("--min-height", type=float, default=0.05, help="T3: ignore floor points below this height (m)")
    ap.add_argument("--max-points", type=int, default=20000)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--max-cam-dist", type=float, default=0.5)
    ap.add_argument("--max-yaw-diff", type=float, default=25.0)
    ap.add_argument("--radius", type=float, default=0.25)
    args = ap.parse_args()

    with open(SCALE_SHIFT_PATH) as f:
        calib = json.load(f)
    print(f"scale_shift: a={calib['scale']:.4f} b={calib['shift']:.4f} | usable={calib.get('usable', 'n/a')} "
          f"| source={calib.get('source', '?')} (validating with --source {args.source})")
    if calib.get("source") not in (None, args.source):
        print("[WARN] --source differs from the source scale_shift.json was fitted on -> results meaningless")

    wanted = {"contact", "floor", "consist"} if args.tests == "all" else set(args.tests.split(","))
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {}
    if "contact" in wanted:
        report["contact"] = test_contact(args, calib)
    if "floor" in wanted:
        report["floor"] = test_floor(args, calib)
    if "consist" in wanted:
        report["consistency"] = test_consistency(args, calib)
    with open(os.path.join(OUT_DIR, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {OUT_DIR}/validation_report.json")


if __name__ == "__main__":
    main()