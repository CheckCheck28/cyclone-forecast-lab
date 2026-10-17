"""IBTrACS ingestion, quality control and basin assignment (global replacement for HURDAT2)."""
import urllib.request
import numpy as np
import pandas as pd
from config import IBTRACS_URL, IBTRACS_CSV, BASINS, STEP_H

COLS = ["SID", "SEASON", "NAME", "ISO_TIME", "NATURE", "LAT", "LON",
        "USA_LAT", "USA_LON", "USA_WIND", "USA_PRES"]


def download_ibtracs(path=IBTRACS_CSV, url=IBTRACS_URL):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"Using local file: {path}")
        return path
    print(f"Downloading IBTrACS (~100 MB) to {path} ...")
    urllib.request.urlretrieve(url, path)
    return path


def geo_basin(lat, lon):
    """Basin index (into config.BASINS) from position. Mirrors the JavaScript app exactly.
    NA/EP split follows the Central-American isthmus; SH basins split at 20E, 135E and 70W."""
    lat = np.asarray(lat, float)
    lon = ((np.asarray(lon, float) + 180.0) % 360.0) - 180.0
    lonb = np.clip(-83.0 - 1.5 * (lat - 8.0), -100.0, -83.0)
    nh = lat >= 0
    conds = [
        nh & (lon < lonb),                 # EP (incl. Central Pacific)
        nh & (lon >= lonb) & (lon <= 20),  # NA
        nh & (lon > 20) & (lon < 100),     # NI
        nh & (lon >= 100),                 # WP
        ~nh & (lon >= -70) & (lon < 20),   # SA
        ~nh & (lon >= 20) & (lon < 135),   # SI
        ~nh & ((lon >= 135) | (lon < -70)),  # SP
    ]
    names = ["EP", "NA", "NI", "WP", "SA", "SI", "SP"]
    idx = np.select(conds, [BASINS.index(n) for n in names], default=0)
    return idx


def load_ibtracs(path=IBTRACS_CSV, min_year=1980):
    """Load, clean and regularise IBTrACS into gap-free 6-hourly track *segments*.

    Fixes vs. the original HURDAT2 notebooks:
      * only synoptic times (00/06/12/18Z) are kept, so every step is exactly 6 h
        (HURDAT2 landfall/off-time records distorted the "per-step" deltas);
      * rows are never silently dropped *inside* a track: a gap starts a new segment,
        so lags/leads never span irregular intervals;
      * one agency's 1-minute winds (USA_WIND: NHC / JTWC) are used for all basins for consistency.
    """
    df = pd.read_csv(path, usecols=COLS, skiprows=[1], na_values=[" ", ""], low_memory=False)
    df["time"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    for c in ["LAT", "LON", "USA_LAT", "USA_LON", "USA_WIND", "USA_PRES"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df[(df["time"].dt.year >= min_year) & (df["NATURE"] == "TS")]
    df = df[(df["time"].dt.hour % STEP_H == 0) & (df["time"].dt.minute == 0)]
    df["lat"] = df["USA_LAT"].fillna(df["LAT"])
    df["lon"] = df["USA_LON"].fillna(df["LON"])
    df["v"] = df["USA_WIND"]
    df["pmin"] = df["USA_PRES"].where((df["USA_PRES"] > 850) & (df["USA_PRES"] < 1040))
    df = df.dropna(subset=["lat", "lon", "v"])
    df = df[(df["v"] >= 10) & (df["v"] <= 200)]
    df = df.rename(columns={"SID": "sid", "NAME": "name"})
    df = df.drop_duplicates(["sid", "time"]).sort_values(["sid", "time"])

    gap = df.groupby("sid")["time"].diff().dt.total_seconds().div(3600).ne(STEP_H)
    df["seg"] = gap.groupby(df["sid"]).cumsum().astype(int)
    df["seg_id"] = df["sid"] + "#" + df["seg"].astype(str)
    df["year0"] = df.groupby("sid")["time"].transform("min").dt.year
    df["basin"] = geo_basin(df["lat"].values, df["lon"].values)
    df["month"] = df["time"].dt.month
    df["doy"] = df["time"].dt.dayofyear
    keep = df.groupby("seg_id")["seg_id"].transform("size") >= 6
    df = df[keep].reset_index(drop=True)
    cols = ["sid", "seg_id", "name", "year0", "time", "lat", "lon", "v", "pmin", "basin", "month", "doy"]
    out = df[cols]
    print(f"Loaded {len(out):,} six-hourly fixes | {out['sid'].nunique():,} storms | "
          f"{out['seg_id'].nunique():,} gap-free segments | {out['time'].dt.year.min()}-{out['time'].dt.year.max()}")
    return out


def fit_pressure_relation(df):
    """Per-basin quadratic wind->pressure fit (pressure is sparse outside the Atlantic/E-Pacific,
    so it is derived from forecast wind rather than being modelled as an independent target)."""
    obs = df.dropna(subset=["pmin"])
    coefs = {}
    glob = np.polyfit(obs["v"], obs["pmin"], 2) if len(obs) > 100 else np.array([0.0, -0.9, 1012.0])
    for i, b in enumerate(BASINS):
        s = obs[obs["basin"] == i]
        coefs[b] = (np.polyfit(s["v"], s["pmin"], 2) if len(s) > 300 else glob).round(6).tolist()
    return coefs
