"""estimate_cam_height.py - Estimate camera height (m) from images, calibrated with depth model.

Idea (camera nhin ngang, pitch ~ 0): diem san o khoang cach Z nam o hang  v = cy + fy*h/Z (+ c neu co pitch).
Ban click diem robot CHAM SAN (giua day than robot) tren ~15-20 frame o nhieu khoang cach, roi fit v theo 1/Z.

B1  python estimate_cam_height.py annotate     # click (can man hinh) -> annotations.csv
    (khong co man hinh: them --dump -> luu anh co thuoc ke hang, tu dien cot v_contact vao annotations.csv)
B2  python estimate_cam_height.py fit          # in h, pitch, khoang tin cay

Phim trong cua so: click = chon diem | Enter/Space = chap nhan | s = bo qua | q = thoat
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

# Vale prrovided by teacher
FY, CY, CX = 725.01303987, 246.52137129, 324.12452038
ROBOTS = ("S1", "S2")


def pick(labels, n, cx, max_off_px=150.0, zmin=0.3, zmax=1.8):
    parts = []
    for rb in ROBOTS:
        d = labels[labels[f"{rb}_valid"] == 1][["img_path", "img_id", f"{rb}_u", f"{rb}_forward_m"]].copy()
        d.columns = ["img_path", "img_id", "u", "forward_m"]
        d["robot"] = rb
        parts.append(d)
    d = pd.concat(parts)
    d = d[(d.forward_m.between(zmin, zmax)) & ((d.u - cx).abs() < max_off_px)].sort_values("forward_m")
    if len(d) == 0:
        raise SystemExit("No suitable marker found (distance 0.3–1.8 m, near the center of the image).")
    idx = np.unique(np.linspace(0, len(d) - 1, min(n, len(d))).astype(int))
    return d.iloc[idx].reset_index(drop=True)


def click(img, title, scale=2):
    pt = {}
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    def on(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pt["p"] = (x / scale, y / scale)
            vis = big.copy()
            cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.imshow(title, vis)

    cv2.imshow(title, big)
    cv2.setMouseCallback(title, on)
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 32) and "p" in pt:
            return pt["p"]
        if k == ord("s"):
            return None
        if k == ord("q"):
            raise KeyboardInterrupt


def cmd_annotate(a):
    L = pd.read_csv(a.labels, dtype={"img_id": str})
    L = L[(L.S1_valid == 1) | (L.S2_valid == 1)]
    P = pick(L, a.n, CX)
    rows = []
    os.makedirs(a.out_dir, exist_ok=True)
    try:
        for k, r in P.iterrows():
            img = cv2.imread(r.img_path)
            if img is None:
                continue
            vis = img.copy()
            same = L[L.img_id == r.img_id].iloc[0]
            for rb in ROBOTS:   # Green robot to chosen robot, red to other robot
                if int(same.get(f"{rb}_valid", 0)):
                    col = (0, 255, 0) if rb == r.robot else (0, 0, 255)
                    uu = int(same[f"{rb}_u"])
                    cv2.line(vis, (uu, 0), (uu, vis.shape[0]), col, 1)
                    cv2.putText(vis, rb, (uu + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            if a.dump:
                for y in range(0, vis.shape[0], 20):
                    cv2.line(vis, (0, y), (vis.shape[1], y), (255, 255, 0) if y % 100 else (0, 255, 255), 1)
                    cv2.putText(vis, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.imwrite(os.path.join(a.out_dir, f"{k:02d}_{r.robot}_{r.img_id}.png"), vis)
                rows.append({**r.to_dict(), "u_contact": np.nan, "v_contact": np.nan})
            else:
                p = click(vis, f"[{k+1}/{len(P)}] {r.robot} Z={r.forward_m:.2f} m - click point on robot touching the ground (or s to skip)")
                if p:
                    rows.append({**r.to_dict(), "u_contact": p[0], "v_contact": p[1]})
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    pd.DataFrame(rows).to_csv(a.annotations, index=False)
    print(f"Saved on {len(rows)} rows -> {a.annotations}" + (" (filled column v_contact)" if a.dump else ""))


def cmd_fit(a):
    D = pd.read_csv(a.annotations).dropna(subset=["v_contact"])
    if len(D) < 6:
        raise SystemExit(f"Only {len(D)} points have v_contact; need >= 6.")
    Z = D.forward_m.to_numpy(float) - a.body_offset       # distance from camera to robot contact point (m)
    y = D.v_contact.to_numpy(float) - CY                  #  pixel row with  cy
    A = np.stack([FY / Z, np.ones_like(Z)], 1)

    def fit(idx):
        sol, *_ = np.linalg.lstsq(A[idx], y[idx], rcond=None)
        return sol

    h, c = fit(np.arange(len(Z)))
    res = y - (A @ np.array([h, c]))
    rng = np.random.default_rng(0)
    bs = np.array([fit(rng.integers(0, len(Z), len(Z))) for _ in range(1000)])
    h0 = float((y * FY / Z).sum() / ((FY / Z) ** 2).sum())   # c = 0 (pitch = 0)
    lo, hi = np.percentile(bs[:, 0], [2.5, 97.5])
    print(f"n={len(Z)}  Z trong [{Z.min():.2f}, {Z.max():.2f}] m")
    print(f"h (affine)   = {h:.3f} m   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"offset c     = {c:+.1f} px  -> pitch ~ {np.degrees(np.arctan(c / FY)):+.1f} deg (0 is the camera is horizontal)")
    print(f"h (pitch=0)  = {h0:.3f} m")
    print(f"RMS phan du  = {np.sqrt(np.mean(res ** 2)):.1f} px")
    if hi - lo > 0.4 * abs(h):
        print("WARNING: wide confidence interval -> add points / increase distance range.")
    if abs(c) > 15:
        print("WARNING: large offset -> camera can be tilted; dy follow the 'horizontal' can be wrong.")
    print(f"\n=> Corrected height: {h:.3f} (contact point = robot ground contact).")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
for name, fn in (("annotate", cmd_annotate), ("fit", cmd_fit)):
    s = sub.add_parser(name)
    s.add_argument("--labels", default="dataset/valid_frames/labels.csv")
    s.add_argument("--annotations", default="dataset/calib/annotations.csv")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--out-dir", default="dataset/calib/annot_dump")
    s.add_argument("--dump", action="store_true")
    s.add_argument("--body-offset", type=float, default=0.0,
                   help="m: distance marker to robot contact point (if known)")
    s.add_argument("--fy", type=float, default=FY)
    s.add_argument("--cy", type=float, default=CY)
    s.add_argument("--cx", type=float, default=CX)
    s.set_defaults(fn=fn)
args = ap.parse_args()
FY, CY, CX = args.fy, args.cy, args.cx   
os.makedirs(os.path.dirname(args.annotations), exist_ok=True)
args.fn(args)