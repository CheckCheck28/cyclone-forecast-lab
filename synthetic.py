"""Synthetic IBTrACS-format generator, ONLY for smoke-testing the pipeline offline
(`python run_pipeline.py --synthetic 900`). Numbers from it mean nothing meteorologically."""
import numpy as np
import pandas as pd

# basin: (weight, lat range, lon range, peak month, hemisphere sign)
SPEC = {"NA": (14, (10, 22), (-60, -20), 9, 1), "EP": (16, (10, 18), (-130, -95), 8, 1),
        "WP": (26, (8, 20), (125, 160), 8, 1), "NI": (4, (8, 16), (68, 90), 10, 1),
        "SI": (14, (-18, -9), (55, 100), 2, -1), "SP": (10, (-18, -9), (160, 200), 2, -1)}


def make(n_storms=900, seed=7, y0=1980, y1=2024):
    rng = np.random.default_rng(seed)
    names, w = list(SPEC), np.array([SPEC[k][0] for k in SPEC], float)
    rows = []
    for i in range(n_storms):
        b = rng.choice(names, p=w / w.sum())
        _, (la0, la1), (lo0, lo1), pk, hs = SPEC[b]
        yr = int(rng.integers(y0, y1 + 1))
        month = int((round(rng.normal(pk, 1.6)) - 1) % 12) + 1
        t = pd.Timestamp(year=yr, month=month, day=int(rng.integers(1, 28)), hour=int(rng.choice([0, 6, 12, 18])))
        lat, lon = rng.uniform(la0, la1), rng.uniform(lo0, lo1)
        v, n = rng.uniform(25, 35), int(rng.integers(18, 70))
        peak = rng.uniform(55, 150); pk_at = rng.uniform(.35, .65) * n
        vx, vy = rng.normal(-14, 3) * (1 if hs > 0 else 1), rng.normal(5, 3) * hs
        sid = f"{yr}{i:04d}N{abs(int(lat)):02d}"
        pts = []
        for k in range(n):
            lat_f = abs(lat)
            gain = peak * np.exp(-((k - pk_at) / (0.32 * n)) ** 2) * np.clip(1.15 - lat_f / 38, 0.15, 1)
            v = 0.7 * v + 0.3 * max(25, gain) + rng.normal(0, 2.2)
            vy += hs * 0.22 + rng.normal(0, 0.9); vx += 0.35 * (lat_f > 24) * 1.6 + rng.normal(0, 1.0)
            vx = np.clip(vx, -30, 30); vy = np.clip(vy, -12 if hs > 0 else -35, 35 if hs > 0 else 12)
            lat += vy * 6 / 111.32; lon += vx * 6 / (111.32 * max(np.cos(np.radians(lat)), .1))
            lon = ((lon + 180) % 360) - 180
            if abs(lat) > 55 or v < 12:
                break
            pts.append((t + pd.Timedelta(hours=6 * k), lat, lon, max(10, round(v / 5) * 5)))
        for j, (tt, la, lo, vv) in enumerate(pts):
            pres = round(1010 - 0.0072 * vv ** 2 - 0.35 * vv + rng.normal(0, 2)) if rng.random() < (0.9 if b in ("NA", "EP") else 0.25) else np.nan
            nat = "TS" if j < len(pts) - 3 else "ET"
            for off in (0, 3):                                      # 3-hourly rows like the real file
                t2 = tt + pd.Timedelta(hours=off)
                rows.append((sid, yr, b, f"STORM{i}", t2.strftime("%Y-%m-%d %H:%M:%S"), nat, la, lo, la, lo, vv, pres))
    cols = ["SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "USA_LAT", "USA_LON", "USA_WIND", "USA_PRES"]
    return pd.DataFrame(rows, columns=cols)


def write_csv(path, **kw):
    df = make(**kw)
    units = pd.DataFrame([[" "] * len(df.columns)], columns=df.columns)       # IBTrACS has a units row
    pd.concat([units, df]).to_csv(path, index=False)       # header, units row, data -- like the real file
    return path
