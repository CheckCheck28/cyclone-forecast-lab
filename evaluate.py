"""Hold-out evaluation: track error (km), intensity MAE/bias (kt) by lead, basin and month,
versus a persistence baseline (constant velocity + constant intensity)."""
import numpy as np
import pandas as pd
from config import BASINS, LEADS
from features import km_to_latlon, haversine_km
from models import predict_raw, LABELS


def score_table(models, test):
    """Long DataFrame: one row per (test origin, lead, model) with errors."""
    raw = predict_raw(models, test["X"])
    raw["base"] = np.zeros_like(raw["gbm"])
    m = test["meta"]
    rows = []
    for name, pred in raw.items():
        dx, dy = pred[:, 0] + test["base"][:, 0], pred[:, 1] + test["base"][:, 1]
        lat_p, lon_p = km_to_latlon(m["lat0"].values, m["lon0"].values, dx, dy)
        v_p = np.clip(m["v0"].values + pred[:, 2], 10, 190)
        rows.append(pd.DataFrame(dict(
            model=name, basin=m["basin"].values, month=m["month"].values, lead=m["lead"].values,
            sid=m["sid"].values, lat0=m["lat0"].values, lon0=m["lon0"].values,
            track_km=haversine_km(m["lat_t"].values, m["lon_t"].values, lat_p, lon_p),
            v_err=v_p - m["v_t"].values)))
    df = pd.concat(rows, ignore_index=True)
    df["v_abs"] = df["v_err"].abs()
    df["hit10"] = (df["v_abs"] <= 10).astype(float)
    return df


def _agg(g):
    return dict(track=round(float(g["track_km"].mean()), 1), mae=round(float(g["v_abs"].mean()), 2),
                bias=round(float(g["v_err"].mean()), 2), hit10=round(float(g["hit10"].mean() * 100), 1),
                n=int(len(g)))


def build_stats(scores, meta_info, order=None, secs=None):
    """Nested dict consumed by the app (and printed as a report)."""
    present = set(scores["model"].unique())
    models = [m for m in (order or ["ens", "gbm", "rf", "base"]) if m in present]
    S = dict(meta=meta_info, leads=LEADS, models=models, labels={m: LABELS[m] for m in models},
             secs=secs or {}, basins=BASINS, overall={}, basin={}, month={}, grid={}, activity=[])

    def series(df):
        out = {}
        for m in models:
            d = df[df["model"] == m]
            per = [_agg(d[d["lead"] == L]) if (d["lead"] == L).any() else None for L in LEADS]
            out[m] = {k: [p[k] if p else None for p in per] for k in ("track", "mae", "bias", "hit10", "n")}
        return out

    S["overall"] = series(scores)
    for bi, b in enumerate(BASINS):
        d = scores[scores["basin"] == bi]
        if len(d) == 0:
            continue
        S["basin"][b] = series(d)
        S["basin"][b]["n_storms"] = int(d["sid"].nunique())
        # month x lead x model:  {model: {lead: {track:[12], mae:[12], n:[12]}}}
        mm = {}
        for m in models:
            dm = d[d["model"] == m]
            mm[m] = {}
            for L in LEADS:
                dl = dm[dm["lead"] == L].groupby("month")
                tr, ma, n = dl["track_km"].mean(), dl["v_abs"].mean(), dl.size()
                mm[m][str(L)] = {"track": [round(float(tr.get(k, np.nan)), 1) if k in tr.index and n[k] >= 5 else None for k in range(1, 13)],
                                 "mae": [round(float(ma.get(k, np.nan)), 2) if k in ma.index and n[k] >= 5 else None for k in range(1, 13)],
                                 "n": [int(n.get(k, 0)) for k in range(1, 13)]}
        S["month"][b] = mm
    # 5-degree error map (ensemble) at several leads
    e = scores[scores["model"] == "ens"]
    for L in (24, 48, 72):
        d = e[e["lead"] == L].copy()
        d["la"] = (np.floor(d["lat0"] / 5) * 5).astype(int)
        d["lo"] = (np.floor(d["lon0"] / 5) * 5).astype(int)
        g = d.groupby(["la", "lo"]).agg(track=("track_km", "mean"), mae=("v_abs", "mean"), n=("track_km", "size")).reset_index()
        g = g[g["n"] >= 15]
        S["grid"][str(L)] = [[int(r.la), int(r.lo), round(r.track, 0), round(r.mae, 1), int(r.n)] for r in g.itertuples()]
    return S


def activity_grid(df):
    """Historical 5-degree activity map (fix counts + mean wind) over the whole record."""
    d = df.assign(la=(np.floor(df["lat"] / 5) * 5).astype(int), lo=(np.floor(df["lon"] / 5) * 5).astype(int))
    g = d.groupby(["la", "lo"]).agg(n=("v", "size"), v=("v", "mean")).reset_index()
    return [[int(r.la), int(r.lo), int(r.n), round(r.v, 1)] for r in g.itertuples() if r.n >= 5]


def print_report(S):
    print("\n=== HOLD-OUT RESULTS (storms from test years, never seen in training) ===")
    for b, d in S["basin"].items():
        e, bs = d["ens"], d["base"]
        i24, i48 = S["leads"].index(24), S["leads"].index(48)
        print(f"{b:>3} | storms {d['n_storms']:>4} | 24h track {e['track'][i24]:>6} km (base {bs['track'][i24]:>6}) | "
              f"48h track {e['track'][i48]:>6} km (base {bs['track'][i48]:>6}) | 24h wind MAE {e['mae'][i24]} kt (base {bs['mae'][i24]})")


def leaderboard(S):
    """Tidy table: one row per (scope, model) with track/wind errors at 24/48/72 h and skill vs persistence."""
    import pandas as pd
    rows, leads = [], S["leads"]
    scopes = [("ALL", S["overall"])] + [(b, S["basin"][b]) for b in S["basin"]]
    for scope, src in scopes:
        base = src["base"]
        for m in S["models"]:
            d, r = src[m], {"scope": scope, "model": m, "label": S["labels"][m]}
            for L in (24, 48, 72):
                i = leads.index(L)
                r[f"track_{L}"], r[f"wind_mae_{L}"] = d["track"][i], d["mae"][i]
            i24, i48 = leads.index(24), leads.index(48)
            r["wind_bias_24"], r["within10kt_24_%"], r["n_24"] = d["bias"][i24], d["hit10"][i24], d["n"][i24]
            r["track_skill_48_%"] = round(100 * (1 - d["track"][i48] / base["track"][i48]), 1)
            r["wind_skill_24_%"] = round(100 * (1 - d["mae"][i24] / base["mae"][i24]), 1)
            r["train_seconds"] = S["secs"].get(m)
            rows.append(r)
    return pd.DataFrame(rows)


def print_leaderboard(lb):
    t = lb[lb["scope"] == "ALL"].sort_values("track_48")
    cols = ["label", "track_24", "track_48", "track_72", "wind_mae_24", "wind_mae_48", "track_skill_48_%", "train_seconds"]
    print("\n=== MODEL LEADERBOARD (all basins, hold-out storms, sorted by 48 h track error) ===")
    print(t[cols].to_string(index=False))
