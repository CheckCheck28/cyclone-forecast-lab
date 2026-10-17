#!/usr/bin/env python
"""End-to-end: download IBTrACS -> clean -> features -> train -> hold-out evaluation -> maps -> app.

    python run_pipeline.py --download            # real global data (needs internet, ~100 MB)
    python run_pipeline.py --data path/to.csv    # already-downloaded IBTrACS CSV
    python run_pipeline.py --synthetic 900 --fast  # offline smoke test with fake data (numbers meaningless)
"""
import argparse, datetime as dt, json
from pathlib import Path
import joblib
import numpy as np

import config as C
from data import download_ibtracs, load_ibtracs, fit_pressure_relation
from features import fit_climatology, build_samples, subset, FEATURES
from models import train_all, GBM_LABEL, resolve_models, available_models, ZOO
from evaluate import score_table, build_stats, activity_grid, print_report, leaderboard, print_leaderboard
from export_app import storm_library, build_bundle, write_app
import maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--synthetic", type=int, default=0, help="generate N fake storms instead of real data")
    ap.add_argument("--fast", action="store_true", help="fewer trees / smaller samples (quick test)")
    ap.add_argument("--models", default="all", help="extra models to benchmark: all | fast (skip slow ones) | none | comma list")
    ap.add_argument("--list-models", action="store_true", help="print available benchmark models and exit")
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--out", type=Path, default=C.OUT_DIR)
    a = ap.parse_args()
    if a.list_models:
        for k in available_models():
            print(f"{k:12s} {ZOO[k]['label']}{'   (slow)' if ZOO[k]['slow'] else ''}")
        return
    a.out.mkdir(parents=True, exist_ok=True)

    if a.synthetic:
        import synthetic
        path = synthetic.write_csv(a.out / "synthetic_ibtracs.csv", n_storms=a.synthetic)
    elif a.data:
        path = a.data
    else:
        path = download_ibtracs() if a.download or not C.IBTRACS_CSV.exists() else C.IBTRACS_CSV

    print("1/6 loading + cleaning");          df = load_ibtracs(path)
    print("2/6 features");
    clim = fit_climatology(df[df["year0"] <= C.TRAIN_END_YEAR])           # training years only -> no leakage
    tab = build_samples(df, clim)
    tr, va, te = (subset(tab, tab["split"] == s) for s in ("train", "val", "test"))
    print(f"    samples  train {len(tr['X']):,} | val {len(va['X']):,} | test {len(te['X']):,}  ({len(FEATURES)} features)")
    print("3/6 training");                     models = train_all(tr, va, fast=a.fast, zoo_keys=resolve_models(a.models))
    print("4/6 hold-out evaluation");          scores = score_table(models, te)
    info = dict(generated=dt.date.today().isoformat(), fixes=int(len(df)), storms=int(df["sid"].nunique()),
                years=[int(df["year0"].min()), int(df["year0"].max())], train_end=C.TRAIN_END_YEAR,
                val_end=C.VAL_END_YEAR, test_from=C.VAL_END_YEAR + 1, gbm=GBM_LABEL, synthetic=bool(a.synthetic))
    stats = build_stats(scores, info, order=models["order"], secs=models["secs"]);  print_report(stats)
    lb = leaderboard(stats); lb.to_csv(a.out / "model_comparison.csv", index=False); print_leaderboard(lb)
    (a.out / "stats.json").write_text(json.dumps(stats))
    joblib.dump(models, a.out / "models.joblib")

    print("5/6 maps + figures")
    act = activity_grid(df)
    if not a.no_maps:
        maps.global_maps(stats, act, a.out); maps.month_heatmaps(stats, a.out, 48); maps.error_growth(stats, a.out); maps.leaderboard_plot(lb, a.out)
        maps.feature_importance(models, FEATURES, a.out); maps.storm_maps(None, df, a.out, C.VAL_END_YEAR + 1)

    print("6/6 browser app")
    pres = fit_pressure_relation(df)
    bundle = build_bundle(models, clim, pres, stats, storm_library(df, C.VAL_END_YEAR + 1), act, info)
    write_app(bundle, Path(__file__).parent / "app" / "hurricane_app.html", a.out / "hurricane_app.html", a.out / "app_data.json")
    print(f"\nDone. Open {a.out / 'hurricane_app.html'} in a browser.")


if __name__ == "__main__":
    main()
