"""
Validate predicted depth using only the measured S1/S2 forward distances in labels.csv.

Uses the same model-loading convention as fit_scale.py:
    from model import ResNetDepth
    weights/best_model.pth

Outputs under dataset/calib/validation_s1_s2/:
    samples.csv          one row per valid S1/S2 measurement
    summary.json         overall and per-robot metrics
    validation_plot.png  ground-truth vs out-of-fold calibrated prediction
    residual_plot.png    out-of-fold error vs ground truth

IMPORTANT:
- Raw model output is not assumed to be meters, so raw MAE/RMSE are not reported.
- Reported calibrated metrics use out-of-fold predictions: each sample is predicted
  by a scale/shift fit that did not train on that sample's temporal block.
- This validates distance at the S1/S2 marker locations only, not dense depth over
  the entire scene or the floor.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS_CSV = Path("dataset/valid_frames/labels.csv")
WEIGHTS_PATH = Path("weights/best_model.pth")
OUT_DIR = Path("dataset/calib/validation_s1_s2")
NET_SIZE = 180
ROBOTS = ("S1", "S2")


def load_model(device):
    import torch
    from model import ResNetDepth

    model = ResNetDepth().to(device)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    # Support checkpoints saved either as a plain state_dict or wrapped dict.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict_one(model, image_path, device):
    import torch

    image = cv2.imread(str(image_path))
    if image is None:
        return None, None

    h0, w0 = image.shape[:2]
    resized = cv2.resize(image, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.from_numpy(rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.no_grad():
        depth = model(tensor).squeeze().detach().cpu().numpy()

    if depth.ndim != 2:
        raise ValueError(f"Unexpected model output shape {depth.shape} for {image_path}")
    return depth.astype(np.float64), (h0, w0)


def extract_samples(labels, model, device, patch_u, patch_v):
    rows = []
    cache = {}
    for _, row in labels.iterrows():
        image_path = str(row["img_path"])
        if image_path not in cache:
            cache[image_path] = predict_one(model, image_path, device)
        depth, original_shape = cache[image_path]
        if depth is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        h0, w0 = original_shape
        dh, dw = depth.shape

        for robot in ROBOTS:
            valid_col = f"{robot}_valid"
            u_col, v_col = f"{robot}_u", f"{robot}_v"
            distance_col = f"{robot}_forward_m"

            required = (valid_col, u_col, v_col, distance_col)
            if any(col not in row.index for col in required):
                raise ValueError(f"labels.csv missing one of these columns: {required}")

            try:
                is_valid = int(row[valid_col]) == 1
            except (TypeError, ValueError):
                is_valid = False
            if not is_valid:
                continue

            u0, v0 = float(row[u_col]), float(row[v_col])
            d_true = float(row[distance_col])
            if not np.isfinite([u0, v0, d_true]).all() or d_true <= 0:
                continue

            # Map coordinates from original RGB image into the model-output grid.
            ui = int(round(u0 * dw / w0))
            vi = int(round(v0 * dh / h0))
            ui = int(np.clip(ui, 0, dw - 1))
            vi = int(np.clip(vi, 0, dh - 1))

            patch = depth[
                max(vi - patch_v, 0):min(vi + patch_v + 1, dh),
                max(ui - patch_u, 0):min(ui + patch_u + 1, dw)
            ]
            patch = patch[np.isfinite(patch) & (patch > 1e-6)]
            if patch.size == 0:
                continue

            q25, med, q75 = np.percentile(patch, [25, 50, 75])
            rows.append({
                "img_id": str(row.get("img_id", "")),
                "csv_idx": int(row["csv_idx"]) if "csv_idx" in row.index and pd.notna(row["csv_idx"]) else len(rows),
                "robot": robot,
                "img_path": image_path,
                "u_rgb": u0,
                "v_rgb": v0,
                "d_true_m": d_true,
                "d_pred_raw": float(med),
                "patch_iqr_raw": float(q75 - q25),
                "patch_n": int(patch.size),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(
            "No usable S1/S2 samples. Check labels.csv paths, valid flags, and column names."
        )
    return result


def fit_scale(x, y):
    # Least-squares fit with zero shift: y ~= a*x
    denom = float(x @ x)
    if denom <= 1e-12:
        raise ValueError("Cannot fit scale: predicted depths are nearly all zero.")
    return float((x @ y) / denom), 0.0


def fit_affine(x, y):
    if len(x) < 2 or np.ptp(x) < 1e-10:
        return fit_scale(x, y)
    A = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    if a <= 0:
        return fit_scale(x, y)
    return float(a), float(b)


def compute_metrics(y, p):
    err = p - y
    denom = np.maximum(np.abs(y), 1e-9)
    return {
        "n": int(len(y)),
        "mae_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "median_absolute_error_m": float(np.median(np.abs(err))),
        "mean_absolute_relative_error_pct": float(np.mean(np.abs(err) / denom) * 100.0),
        "median_absolute_relative_error_pct": float(np.median(np.abs(err) / denom) * 100.0),
        "bias_m": float(np.mean(err)),
    }


def add_blocked_cv_predictions(samples, folds):
    # Keep frames ordered by csv_idx; split into contiguous blocks to reduce
    # optimistic leakage from adjacent, visually similar frames.
    samples = samples.sort_values(["csv_idx", "robot"]).reset_index(drop=True)
    n = len(samples)
    k = max(2, min(int(folds), n // 3))
    if n < 6:
        raise RuntimeError(f"Only {n} usable samples; need at least 6 for blocked CV.")
    order = np.arange(n)
    blocks = np.array_split(order, k)
    samples["pred_scale_cv_m"] = np.nan
    samples["pred_affine_cv_m"] = np.nan
    samples["cv_fold"] = -1

    x_all = samples["d_pred_raw"].to_numpy(float)
    y_all = samples["d_true_m"].to_numpy(float)

    for fold_id, test_idx in enumerate(blocks):
        train_idx = np.setdiff1d(order, test_idx)
        if len(train_idx) < 2:
            continue
        a_s, b_s = fit_scale(x_all[train_idx], y_all[train_idx])
        a_a, b_a = fit_affine(x_all[train_idx], y_all[train_idx])
        samples.loc[test_idx, "pred_scale_cv_m"] = a_s * x_all[test_idx] + b_s
        samples.loc[test_idx, "pred_affine_cv_m"] = a_a * x_all[test_idx] + b_a
        samples.loc[test_idx, "cv_fold"] = fold_id

    return samples


def main():
    global NET_SIZE, WEIGHTS_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS_CSV)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--net-size", type=int, default=NET_SIZE)
    parser.add_argument("--patch-u", type=int, default=3, help="Half-width of marker patch in model-output pixels.")
    parser.add_argument("--patch-v", type=int, default=6, help="Half-height of marker patch in model-output pixels.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    NET_SIZE = args.net_size
    WEIGHTS_PATH = args.weights

    if not args.labels.exists():
        raise SystemExit(f"labels.csv not found: {args.labels}")
    if not args.weights.exists():
        raise SystemExit(f"Model weights not found: {args.weights}")

    labels = pd.read_csv(args.labels, dtype={"img_id": str})
    needed = {"img_path", "S1_valid", "S1_u", "S1_v", "S1_forward_m",
              "S2_valid", "S2_u", "S2_v", "S2_forward_m"}
    missing = needed - set(labels.columns)
    if missing:
        raise SystemExit(f"labels.csv is missing columns: {sorted(missing)}")

    labels = labels[
        (pd.to_numeric(labels["S1_valid"], errors="coerce").fillna(0) == 1)
        | (pd.to_numeric(labels["S2_valid"], errors="coerce").fillna(0) == 1)
    ].reset_index(drop=True)
    print(f"Frames with at least one valid robot marker: {len(labels)}")

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            raise SystemExit("PyTorch is required. Activate the project's environment first.")
    else:
        device = args.device
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("You selected --device cuda, but CUDA is unavailable.")

    print(f"Loading model on {device}: {args.weights}")
    model = load_model(device)
    samples = extract_samples(labels, model, device, args.patch_u, args.patch_v)
    samples = add_blocked_cv_predictions(samples, args.folds)

    summary = {
        "model_weights": str(args.weights),
        "labels_csv": str(args.labels),
        "note": "Only S1/S2 marker forward distances are ground truth; this does not validate dense scene depth or floor geometry.",
        "n_frames_with_valid_marker": int(len(labels)),
        "n_samples": int(len(samples)),
        "raw_output_range": [
            float(samples["d_pred_raw"].min()), float(samples["d_pred_raw"].max())
        ],
        "ground_truth_range_m": [
            float(samples["d_true_m"].min()), float(samples["d_true_m"].max())
        ],
        "overall": {},
        "per_robot": {},
    }

    for model_name, col in (
        ("scale_only_blocked_cv", "pred_scale_cv_m"),
        ("affine_blocked_cv", "pred_affine_cv_m"),
    ):
        valid = np.isfinite(samples[col].to_numpy(float))
        summary["overall"][model_name] = compute_metrics(
            samples.loc[valid, "d_true_m"].to_numpy(float),
            samples.loc[valid, col].to_numpy(float),
        )
        summary["per_robot"][model_name] = {}
        for robot in ROBOTS:
            mask = (samples["robot"] == robot) & valid
            if mask.sum():
                summary["per_robot"][model_name][robot] = compute_metrics(
                    samples.loc[mask, "d_true_m"].to_numpy(float),
                    samples.loc[mask, col].to_numpy(float),
                )

    # Fit final coefficients on all samples for future use, but do not use
    # these in-sample predictions as the headline validation metrics.
    x = samples["d_pred_raw"].to_numpy(float)
    y = samples["d_true_m"].to_numpy(float)
    a_s, b_s = fit_scale(x, y)
    a_a, b_a = fit_affine(x, y)
    summary["fit_all_data_coefficients_for_reference_only"] = {
        "scale_only": {"scale": a_s, "shift": b_s},
        "affine": {"scale": a_a, "shift": b_a},
        "warning": "These coefficients are fit on all samples; use blocked-CV metrics above for validation."
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.out_dir / "samples.csv", index=False)
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Ground truth vs out-of-fold prediction.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(samples["d_true_m"], samples["pred_scale_cv_m"], s=22, alpha=0.75, label="Scale-only CV")
    ax.scatter(samples["d_true_m"], samples["pred_affine_cv_m"], s=22, alpha=0.75, label="Affine CV")
    low = min(samples["d_true_m"].min(), samples["pred_scale_cv_m"].min(), samples["pred_affine_cv_m"].min())
    high = max(samples["d_true_m"].max(), samples["pred_scale_cv_m"].max(), samples["pred_affine_cv_m"].max())
    ax.plot([low, high], [low, high], "k--", label="Ideal: prediction = ground truth")
    ax.set_xlabel("Ground-truth S1/S2 forward distance (m)")
    ax.set_ylabel("Out-of-fold calibrated prediction (m)")
    ax.set_title("S1/S2 distance validation (blocked cross-validation)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "validation_plot.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for col, label in (
        ("pred_scale_cv_m", "Scale-only CV"),
        ("pred_affine_cv_m", "Affine CV"),
    ):
        residual = samples[col] - samples["d_true_m"]
        ax.scatter(samples["d_true_m"], residual, s=22, alpha=0.7, label=label)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Ground-truth distance (m)")
    ax.set_ylabel("Prediction error (m)")
    ax.set_title("Out-of-fold residuals")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "residual_plot.png", dpi=160)
    plt.close(fig)

    print("\n=== BLOCKED-CV VALIDATION (headline metrics) ===")
    for name, metrics in summary["overall"].items():
        print(f"\n{name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"\nSaved results to: {args.out_dir.resolve()}")
    print("  samples.csv")
    print("  summary.json")
    print("  validation_plot.png")
    print("  residual_plot.png")
    print("\nReminder: this validates distances at S1/S2 markers only, not the floor or dense depth map.")


if __name__ == "__main__":
    main()