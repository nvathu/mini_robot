
import json
import os

import numpy as np

CLICKS_JSON = "dataset/calib/camera_height.json"
INTRINSICS_PATH = "dataset/calib/intrinsics.json"
OUT_JSON = "dataset/calib/floor_geometry.json"


def theil_sen(x, y):
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-9
    slope = np.median((y[j] - y[i])[ok] / dx[ok])
    return float(slope), float(np.median(y - slope * x))


def fit_models(v, z, fy, cy):
    x = fy / z                                   # px per metre of camera height
    h_a = float(np.median((v - cy) / x))         # model A
    res_a = v - (cy + h_a * x)
    h_b, vh_b = theil_sen(x, v)                  # model B
    res_b = v - (vh_b + h_b * x)
    mad = lambda r: float(np.median(np.abs(r - np.median(r))))
    return {"A": (h_a, cy, mad(res_a)), "B": (h_b, vh_b, mad(res_b))}


def main():
    with open(CLICKS_JSON) as f:
        S = json.load(f)["samples"]
    with open(INTRINSICS_PATH) as f:
        K = json.load(f)
    v = np.array([s["v_click"] for s in S])
    z = np.array([s["d_true"] for s in S])
    fy, cy = K["fy"], K["cy"]
    print(f"{len(S)} clicks | Z {z.min():.2f}-{z.max():.2f} m | fy={fy:.1f} cy={cy:.1f}")

    m = fit_models(v, z, fy, cy)
    print(f"\nA (zero pitch, v_h=cy): h={m['A'][0]:.3f} m | residual MAD {m['A'][2]:.1f} px")
    print(f"B (free horizon)      : h={m['B'][0]:.3f} m, v_h={m['B'][1]:.1f} px | residual MAD {m['B'][2]:.1f} px")

    rng = np.random.default_rng(0)
    hs, vhs = [], []
    for _ in range(1000):
        k = rng.integers(0, len(v), len(v))
        if np.ptp(z[k]) < 0.2:
            continue
        h, vh = theil_sen(fy / z[k], v[k])
        hs.append(h); vhs.append(vh)
    h_lo, h_hi = np.percentile(hs, [2.5, 97.5])
    vh_lo, vh_hi = np.percentile(vhs, [2.5, 97.5])
    delta = m["B"][1] - cy
    print(f"B bootstrap 95% CI: h [{h_lo:.3f}, {h_hi:.3f}] m | v_h [{vh_lo:.1f}, {vh_hi:.1f}] px")
    print(f"Horizon offset vs cy: {delta:+.1f} px (= {np.degrees(np.arctan(delta / fy)):+.2f} deg; "
          f"+ means horizon below cy: camera pitched up or true cy larger than assumed)")

    use_b = not (vh_lo <= cy <= vh_hi) or m["B"][2] < 0.8 * m["A"][2]
    sel = "B" if use_b else "A"
    h, vh, _ = m[sel]
    print(f"\nSelected model {sel}: h={h:.3f} m, v_h={vh:.1f} px "
          f"({'horizon offset is significant or clearly lowers residual' if use_b else 'zero-pitch model is adequate'})")
    print(f"Sensitivity: row error 1 deg = {fy * np.tan(np.radians(1)):.1f} px; at Z=1.7 m floor is only {fy * h / 1.7:.0f} px below horizon")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"model": sel, "h_m": h, "v_horizon_px": vh, "h_ci95_m": [float(h_lo), float(h_hi)],
                   "v_h_ci95_px": [float(vh_lo), float(vh_hi)], "n": len(S)}, f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()