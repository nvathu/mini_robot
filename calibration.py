"""export_calibration.py - Fit affine (a, b) tren cac diem cham san do BAN CLICK va ghi scale_shift.json.
Cung schema voi fit_scale.py (selected, scale, shift, d_true_range, d_pred_range, ...) nen 3d_generation.py /
point_could_each.py dung duoc khong can sua. Luu file cu thanh scale_shift.old.json.

  python export_calibration.py --body-offset 0.07
"""
import argparse
import json
import os
import shutil

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_scale import fit_affine, fit_scale_only

NET = 180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="dataset/calib/annotations.csv")
    ap.add_argument("--weights", default="weights/best_model.pth")
    ap.add_argument("--out-dir", default="dataset/calib")
    ap.add_argument("--body-offset", type=float, default=0.07, help="m; PHAI khop voi so ban dung khi validate; hoi thay kich thuoc robot")
    ap.add_argument("--patch-u", type=int, default=2)
    ap.add_argument("--patch-v", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.15)
    args = ap.parse_args()

    D = pd.read_csv(args.annotations, dtype={"img_id": str}).dropna(subset=["u_contact", "v_contact"])
    D = D.sort_values("img_id").reset_index(drop=True)

    import torch
    from model import ResNetDepth
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetDepth().to(dev)
    model.load_state_dict(torch.load(args.weights, map_location=dev))
    model.eval()

    cache, xs, ys = {}, [], []
    for _, r in D.iterrows():
        p = r.img_path
        if p not in cache:
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            x = cv2.cvtColor(cv2.resize(bgr, (NET, NET), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(dev)
            with torch.inference_mode():
                cache[p] = (model(t)[0, 0].cpu().numpy(), bgr.shape[:2])    # head = none
        d, (h0, w0) = cache[p]
        ui, vi = int(round(r.u_contact * NET / w0)), int(round(r.v_contact * NET / h0))
        z = r.forward_m - args.body_offset
        if not (0 <= ui < NET and 0 <= vi < NET) or z <= 0.1:
            continue
        patch = d[max(vi - args.patch_v, 0):vi + args.patch_v + 1, max(ui - args.patch_u, 0):ui + args.patch_u + 1]
        patch = patch[np.isfinite(patch) & (patch > 1e-3)]
        if patch.size >= 0.6 * (2 * args.patch_u + 1) * (2 * args.patch_v + 1):
            xs.append(float(np.median(patch))); ys.append(z)
    x, y = np.array(xs), np.array(ys)
    n = len(y)
    if n < 20:
        raise SystemExit(f"Chi {n} mau hop le; can >= 20.")

    a_a, b_a, inl_a = fit_affine(x, y, args.thr)
    a_s, b_s, inl_s = fit_scale_only(x, y, args.thr)
    if a_a <= 0:
        raise SystemExit(f"He so a={a_a:.4f} <= 0: calibration khong hop le.")

    # held-out theo block thoi gian (chi de bao cao)
    blocks = np.array_split(np.arange(n), min(args.folds, max(2, n // 4)))
    oof = np.full(n, np.nan)
    for te in blocks:
        tr = np.setdiff1d(np.arange(n), te)
        a, b, _ = fit_affine(x[tr], y[tr], args.thr)
        oof[te] = a * x[te] + b
    e = oof - y
    cv = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
          "median_rel_err": float(np.median(np.abs(e) / y))}

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "scale_shift.json")
    if os.path.exists(path):
        shutil.copy(path, os.path.join(args.out_dir, "scale_shift.old.json"))
    out = {
        "selected": "affine", "scale": float(a_a), "shift": float(b_a),
        "scale_only": {"a": float(a_s), "b": 0.0, "n_inliers": int(inl_s.sum())},
        "affine": {"a": float(a_a), "b": float(b_a), "n_inliers": int(inl_a.sum()), "cv": cv},
        "n_samples": int(n), "cv_folds": len(blocks),
        "d_true_range": [float(y.min()), float(y.max())],
        "d_pred_range": [float(x.min()), float(x.max())],
        "inlier_thr_m": args.thr, "patch_half_uv": [args.patch_u, args.patch_v],
        "head": "none (output raw cua ResNetDepth, khong qua activation)",
        "calibration_source": "pixel cham san do nguoi click (estimate_cam_height.py annotate), ground truth = forward_m - body_offset",
        "body_offset_m": args.body_offset,
        "validated_range_m": [float(y.min()), float(y.max())],
        "warnings": [
            "Chi validate tai DIEM SAN duoi robot, 1 session; khong chung minh cho vat the/tuong.",
            "Ngoai validated_range_m la ngoai suy, chua kiem chung.",
            "Sai so tuyet doi ~0.10 m bi chan boi nhieu click (khong biet student tot hon muc nay hay khong).",
            "body_offset sai lam b lech mot hang so (~ do chenh offset).",
            "Intrinsics trong 3d_generation.py/intrinsics.json phai cap nhat theo calibration chinh thuc cua thay; chua undistort anh.",
        ],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=16, label="diem click")
    xs_ = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs_, a_a * xs_ + b_a, "b-", label=f"affine a={a_a:.3f}, b={b_a:.3f}")
    ax.set_xlabel("d_pred (raw)"); ax.set_ylabel("Z_contact (m)"); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "scale_shift_check.png"), dpi=140)

    print(f"n={n} | affine: a={a_a:.4f}, b={b_a:.4f} | scale_only: a={a_s:.4f}")
    print(f"held-out: MAE={cv['mae']:.3f} m | rel trung vi={cv['median_rel_err'] * 100:.1f}% | bias={cv['bias']:+.3f} m")
    print(f"validated_range = [{y.min():.2f}, {y.max():.2f}] m | d_pred = [{x.min():.3f}, {x.max():.3f}]")
    print(f"Da ghi {path} (file cu -> scale_shift.old.json)")


if __name__ == "__main__":
    main()