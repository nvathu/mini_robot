"""verify_clicked.py - Validate depth student tai cac pixel do BAN CLICK (vi tri lay mau KHONG tinh tu Z -> khong ro ri).

Dau vao: annotations.csv tu `estimate_cam_height.py annotate` (cot img_path, img_id, robot, forward_m, u_contact, v_contact).
So sanh tren cung cac pixel: student (fit affine held-out), mean-map prior, baseline hang so.

  python estimate_cam_height.py annotate --n 60        # click them (>= 50 diem)
  python verify_clicked.py --body-offset 0.07
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from fit_scale import fit_affine

NET = 180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="dataset/calib/annotations.csv")
    ap.add_argument("--labels", default="dataset/valid_frames/labels.csv")
    ap.add_argument("--weights", default="weights/best_model.pth")
    ap.add_argument("--body-offset", type=float, default=0.07)
    ap.add_argument("--patch-u", type=int, default=2)
    ap.add_argument("--patch-v", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.15)
    args = ap.parse_args()

    D = pd.read_csv(args.annotations, dtype={"img_id": str}).dropna(subset=["u_contact", "v_contact"])
    D = D.sort_values("img_id").reset_index(drop=True)       # img_id ~ timestamp -> block theo thoi gian
    if len(D) < 10:
        raise SystemExit(f"Chi co {len(D)} diem click; can >= 10 (khuyen nghi >= 50).")

    import torch
    from model import ResNetDepth
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetDepth().to(dev)
    model.load_state_dict(torch.load(args.weights, map_location=dev))
    model.eval()

    L = pd.read_csv(args.labels, dtype={"img_id": str})
    L = L[(L.S1_valid == 1) | (L.S2_valid == 1)]
    paths = set(L["img_path"]) | set(D["img_path"])
    depths = {}
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        x = cv2.cvtColor(cv2.resize(bgr, (NET, NET), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(dev)
        with torch.inference_mode():
            depths[p] = (model(t)[0, 0].cpu().numpy(), bgr.shape[:2])
    mean_map = np.mean([v[0] for v in depths.values()], axis=0)

    def sample(d, u, v):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < NET and 0 <= vi < NET):
            return np.nan
        patch = d[max(vi - args.patch_v, 0):vi + args.patch_v + 1, max(ui - args.patch_u, 0):ui + args.patch_u + 1]
        patch = patch[np.isfinite(patch) & (patch > 1e-3)]
        return float(np.median(patch)) if patch.size >= 0.6 * (2 * args.patch_u + 1) * (2 * args.patch_v + 1) else np.nan

    xs, x0, ys, keep = [], [], [], []
    for i, r in D.iterrows():
        if r.img_path not in depths:
            continue
        d, (h0, w0) = depths[r.img_path]
        u, v = r.u_contact * NET / w0, r.v_contact * NET / h0
        s, s0 = sample(d, u, v), sample(mean_map, u, v)
        if np.isfinite(s) and np.isfinite(s0) and r.forward_m - args.body_offset > 0.1:
            xs.append(s); x0.append(s0); ys.append(r.forward_m - args.body_offset); keep.append(i)
    X, X0, y = map(np.array, (xs, x0, ys))
    n = len(y)
    print(f"diem hop le={n}/{len(D)} | Z_contact trong [{y.min():.2f}, {y.max():.2f}] m | body_offset={args.body_offset}")

    blocks = np.array_split(np.arange(n), min(args.folds, max(2, n // 4)))
    P, P0, B = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for te in blocks:
        tr = np.setdiff1d(np.arange(n), te)
        a, b, _ = fit_affine(X[tr], y[tr], args.thr);   P[te] = a * X[te] + b
        a, b, _ = fit_affine(X0[tr], y[tr], args.thr);  P0[te] = a * X0[te] + b
        B[te] = np.median(y[tr])

    def rep(name, p):
        e = p - y
        sp = pd.Series(p).rank().corr(pd.Series(y).rank())
        print(f"{name:<22} MAE={np.abs(e).mean():.3f} m | rel trung vi={np.median(np.abs(e) / y) * 100:5.1f}% | "
              f"bias={e.mean():+.3f} | Spearman={sp:+.2f}")

    print("\n=== HELD-OUT theo block thoi gian (pixel tu CLICK, khong phu thuoc Z) ===")
    rep("Student", P); rep("Mean-map prior", P0); rep("Baseline hang so", B)
    ms, mp = np.abs(P - y).mean(), np.abs(P0 - y).mean()
    print("-" * 70)
    if ms < 0.8 * mp:
        print("=> Student tot hon prior hinh hoc san: co thong tin ngoai vi tri hang.")
    elif ms <= 1.2 * mp:
        print("=> Student ~ prior hinh hoc san: nhat quan voi hinh hoc san, CHUA chung minh them gi.")
    else:
        print("=> Student KEM hon prior hinh hoc san tai cac diem nay.")
    if n < 40:
        print(f"LUU Y: n={n} nho -> chi mang tinh chi bao. Click them (>= 50-60 diem).")
    print("Gioi han: chi do tai DIEM SAN duoi robot; khong chung minh cho vat the/tuong/depth toan anh.")


if __name__ == "__main__":
    main()
