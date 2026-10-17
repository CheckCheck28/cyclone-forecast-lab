"""Feature engineering. ONE function (`compute_features`) is used for training, evaluation and the
browser app (re-implemented line-for-line in JavaScript) so there is no train/serve skew."""
import numpy as np
import pandas as pd
from data import geo_basin
from config import (BASINS, LEADS, STEP_H, KM_PER_DEG, PRES_BAND_DEG, HIST_STEPS,
                    TRAIN_END_YEAR, VAL_END_YEAR)

N_BANDS = 10
BASE_FEATURES = (
    ["lat", "abs_lat", "hemi", "sin_lon", "cos_lon", "lon360"]
    + [f"b_{b}" for b in BASINS]
    + ["v", "dv6", "dv12", "dv24", "e6", "n6", "e12", "n12", "e24", "n24",
       "sin_season", "cos_season", "clim_p95", "pot_gap"]
)
FEATURES = BASE_FEATURES + ["lead"]


# ----------------------------------------------------------------------------- geometry
def disp_km(lat1, lon1, lat2, lon2):
    """East/north displacement (km) from point 1 to point 2, dateline-safe."""
    dlon = ((np.asarray(lon2, float) - np.asarray(lon1, float) + 180.0) % 360.0) - 180.0
    dx = dlon * KM_PER_DEG * np.cos(np.radians((np.asarray(lat1, float) + np.asarray(lat2, float)) / 2))
    dy = (np.asarray(lat2, float) - np.asarray(lat1, float)) * KM_PER_DEG
    return dx, dy


def km_to_latlon(lat, lon, dx, dy):
    lat2 = np.asarray(lat, float) + np.asarray(dy, float) / KM_PER_DEG
    cosv = np.maximum(np.cos(np.radians((np.asarray(lat, float) + lat2) / 2)), 0.05)
    lon2 = np.asarray(lon, float) + np.asarray(dx, float) / (KM_PER_DEG * cosv)
    return lat2, ((lon2 + 180.0) % 360.0) - 180.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1))


# ----------------------------------------------------------------------------- climatology
def fit_climatology(df, q=0.95, min_n=30):
    """Empirical 95th-percentile intensity per (basin, month, 5-deg latitude band) from TRAINING data only.
    Acts as a data-driven proxy for maximum potential intensity (SST/shear are not available globally)."""
    band = np.minimum((df["lat"].abs() // PRES_BAND_DEG).astype(int), N_BANDS - 1)
    g = df.assign(band=band)
    glob = g["v"].quantile(q)
    table = np.full((len(BASINS), 12, N_BANDS), np.nan)
    basin_q = g.groupby("basin")["v"].quantile(q)
    for (b, m, bd), s in g.groupby(["basin", "month", "band"])["v"]:
        if len(s) >= min_n:
            table[b, m - 1, bd] = s.quantile(q)
    for b in range(len(BASINS)):
        fill = basin_q.get(b, glob)
        t = table[b]
        t[np.isnan(t)] = fill
    return np.round(table, 1)


# ----------------------------------------------------------------------------- core features
def compute_features(lat, lon, v, lat6, lon6, v6, lat12, lon12, v12, lat24, lon24, v24,
                     month, doy, clim):
    """Vectorised feature builder (arrays or scalars). Returns (X[n, len(BASE_FEATURES)], basin_idx)."""
    lat, lon, v = (np.atleast_1d(np.asarray(a, float)) for a in (lat, lon, v))
    lat6, lon6, v6, lat12, lon12, v12, lat24, lon24, v24 = (
        np.atleast_1d(np.asarray(a, float)) for a in (lat6, lon6, v6, lat12, lon12, v12, lat24, lon24, v24))
    month = np.atleast_1d(np.asarray(month, int)); doy = np.atleast_1d(np.asarray(doy, float))
    basin = geo_basin(lat, lon)
    e6, n6 = (a / 6.0 for a in disp_km(lat6, lon6, lat, lon))
    e12, n12 = (a / 12.0 for a in disp_km(lat12, lon12, lat, lon))
    e24, n24 = (a / 24.0 for a in disp_km(lat24, lon24, lat, lon))
    phase = np.where(lat >= 0, doy, (doy + 210.0) % 365.0)       # align SH season peak (Feb) with NH (Sep)
    band = np.minimum((np.abs(lat) // PRES_BAND_DEG).astype(int), N_BANDS - 1)
    clim_v = np.asarray(clim)[basin, month - 1, band]
    cols = [lat, np.abs(lat), (lat >= 0).astype(float), np.sin(np.radians(lon)), np.cos(np.radians(lon)), lon % 360.0]
    cols += [(basin == i).astype(float) for i in range(len(BASINS))]
    cols += [v, v - v6, v - v12, v - v24, e6, n6, e12, n12, e24, n24,
             np.sin(2 * np.pi * phase / 365.25), np.cos(2 * np.pi * phase / 365.25), clim_v, clim_v - v]
    return np.column_stack(cols), basin


# ----------------------------------------------------------------------------- tables
def split_label(year0):
    return np.where(year0 <= TRAIN_END_YEAR, "train", np.where(year0 <= VAL_END_YEAR, "val", "test"))


def build_samples(df, clim, leads=LEADS):
    """Stack one row per (forecast origin, lead). Origins need 24 h of history and a contiguous future."""
    df = df.sort_values(["seg_id", "time"]).reset_index(drop=True)
    g = df.groupby("seg_id")
    lag = {k: {c: g[c].shift(k) for c in ("lat", "lon", "v")} for k in HIST_STEPS}
    ok = np.ones(len(df), bool)
    for k in HIST_STEPS:
        ok &= lag[k]["lat"].notna().values
    Xb, basin = compute_features(
        df["lat"], df["lon"], df["v"],
        lag[1]["lat"], lag[1]["lon"], lag[1]["v"], lag[2]["lat"], lag[2]["lon"], lag[2]["v"],
        lag[4]["lat"], lag[4]["lon"], lag[4]["v"], df["month"], df["doy"], clim)
    e12, n12 = Xb[:, BASE_FEATURES.index("e12")], Xb[:, BASE_FEATURES.index("n12")]
    parts = []
    for L in leads:
        k = L // STEP_H
        fl, fo, fv = (g[c].shift(-k) for c in ("lat", "lon", "v"))
        m = ok & fl.notna().values
        if not m.any():
            continue
        dx, dy = disp_km(df["lat"][m], df["lon"][m], fl[m], fo[m])
        base_x, base_y = L * e12[m], L * n12[m]
        X = np.column_stack([Xb[m], np.full(m.sum(), L)])
        parts.append(dict(
            X=X.astype(np.float32),
            y=np.column_stack([dx - base_x, dy - base_y, fv[m].values - df["v"][m].values]).astype(np.float32),
            base=np.column_stack([base_x, base_y]),
            meta=pd.DataFrame(dict(
                seg_id=df["seg_id"][m].values, sid=df["sid"][m].values, year0=df["year0"][m].values,
                month=df["month"][m].values, basin=basin[m], lead=L, time=df["time"][m].values,
                lat0=df["lat"][m].values, lon0=df["lon"][m].values, v0=df["v"][m].values,
                lat_t=fl[m].values, lon_t=fo[m].values, v_t=fv[m].values))))
    out = dict(X=np.vstack([p["X"] for p in parts]), y=np.vstack([p["y"] for p in parts]),
               base=np.vstack([p["base"] for p in parts]),
               meta=pd.concat([p["meta"] for p in parts], ignore_index=True))
    out["split"] = split_label(out["meta"]["year0"].values)
    return out


def subset(tab, mask):
    return dict(X=tab["X"][mask], y=tab["y"][mask], base=tab["base"][mask],
                meta=tab["meta"][mask].reset_index(drop=True), split=tab["split"][mask])
