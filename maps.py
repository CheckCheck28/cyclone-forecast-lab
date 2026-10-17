"""Static map + statistics figures (matplotlib; coastlines are drawn if `cartopy` is installed)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import BASINS, BASIN_NAMES, BASIN_BOX, LEADS

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

SS_COLORS = [(0, "#5ec8e5"), (34, "#2fa36b"), (64, "#f2c744"), (83, "#f39a2b"), (96, "#e8602c"), (113, "#c8201f"), (137, "#8e1b5e")]


def _ax(fig, pos, extent=None):
    if HAS_CARTOPY:
        ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree(central_longitude=180))
        ax.add_feature(cfeature.LAND, facecolor="#e6e2d8"); ax.add_feature(cfeature.COASTLINE, linewidth=.4)
        ax.gridlines(draw_labels=False, linewidth=.3, alpha=.5)
    else:
        ax = fig.add_subplot(*pos); ax.grid(alpha=.3)
    return ax


def _scatter(ax, lo, la, **kw):
    if HAS_CARTOPY:
        kw["transform"] = ccrs.PlateCarree()
        return ax.scatter(lo, la, **kw)
    return ax.scatter((np.asarray(lo) % 360), la, **kw)           # 0-360 so Pacific is contiguous


def _cells(ax, cells, vi, cmap, label, fig, vmin=None, vmax=None):
    la = np.array([c[0] for c in cells]) + 2.5; lo = np.array([c[1] for c in cells]) + 2.5
    v = np.array([c[vi] for c in cells], float)
    sc = _scatter(ax, lo, la, c=v, s=48, marker="s", cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    fig.colorbar(sc, ax=ax, shrink=.6, label=label)
    ax.set_ylim(-45, 55)
    if not HAS_CARTOPY:
        ax.set_xlim(20, 380)


def global_maps(stats, activity, out):
    fig = plt.figure(figsize=(16, 13))
    ax = _ax(fig, (3, 1, 1)); _cells(ax, activity, 2, "magma_r", "6-hourly fixes per 5° cell", fig)
    ax.set_title("Historical cyclone activity (IBTrACS, 1980-present)")
    for i, L in enumerate((24, 72)):
        ax = _ax(fig, (3, 1, 2 + i)); _cells(ax, stats["grid"][str(L)], 2, "viridis_r", "Mean track error (km)", fig)
        ax.set_title(f"Ensemble track error on unseen storms, +{L} h (by forecast-origin cell, n>=15)")
    fig.tight_layout(); fig.savefig(out / "map_global_activity_error.png", dpi=130); plt.close(fig)


def month_heatmaps(stats, out, lead=48):
    bs = [b for b in BASINS if b in stats["month"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    for ax, key, title, cmap in ((axes[0], "track", f"Track error (km) at +{lead} h", "YlOrRd"),
                                 (axes[1], "mae", f"Wind MAE (kt) at +{lead} h", "PuBu")):
        M = np.array([[np.nan if x is None else x for x in stats["month"][b]["ens"][str(lead)][key]] for b in bs], float)
        im = ax.imshow(M, aspect="auto", cmap=cmap)
        ax.set_xticks(range(12)); ax.set_xticklabels(list("JFMAMJJASOND")); ax.set_yticks(range(len(bs))); ax.set_yticklabels(bs)
        for i in range(M.shape[0]):
            for j in range(12):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7)
        ax.set_title(title); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out / "accuracy_by_basin_month.png", dpi=130); plt.close(fig)


def error_growth(stats, out, top=6):
    """Error vs lead for the ensemble, the persistence baseline and the `top` best other models."""
    i48 = stats["leads"].index(48)
    ranked = sorted([m for m in stats["models"] if m not in ("base", "ens")], key=lambda m: stats["overall"][m]["track"][i48] or 1e9)
    show = ["ens"] + ranked[:top] + ["base"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m in show:
        st = "k-." if m == "base" else ("-o" if m == "ens" else "--")
        axes[0].plot(LEADS, stats["overall"][m]["track"], st, label=stats["labels"][m])
        axes[1].plot(LEADS, stats["overall"][m]["mae"], st, label=stats["labels"][m])
    axes[0].set(title="Track error vs lead time (all basins, unseen storms)", xlabel="lead (h)", ylabel="km")
    axes[1].set(title="Intensity MAE vs lead time", xlabel="lead (h)", ylabel="kt")
    axes[0].legend(fontsize=8); [a.grid(alpha=.3) for a in axes]
    fig.tight_layout(); fig.savefig(out / "error_growth.png", dpi=130); plt.close(fig)


def leaderboard_plot(lb, out):
    """Horizontal bars: every model's 48 h track error and 24 h wind MAE (all basins), baseline marked."""
    d = lb[lb["scope"] == "ALL"].sort_values("track_48", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.42 * len(d) + 1.5)))
    for ax, col, ttl in ((axes[0], "track_48", "Track error at +48 h (km), lower is better"),
                         (axes[1], "wind_mae_24", "Wind MAE at +24 h (kt), lower is better")):
        colors = ["#7a8793" if m == "base" else "#0b7285" if m in ("ens", "stack") else "#e8602c" for m in d["model"]]
        ax.barh(d["label"], d[col], color=colors)
        base = d.loc[d["model"] == "base", col]
        if len(base):
            ax.axvline(float(base.iloc[0]), color="k", ls=":", lw=1)
        ax.set_title(ttl, fontsize=10); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "model_leaderboard.png", dpi=130); plt.close(fig)


def feature_importance(models, features, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, j, t in zip(axes, range(3), ("Δ east (km)", "Δ north (km)", "Δ wind (kt)")):
        m = models["rf"][j]
        imp = m.feature_importances_
        o = np.argsort(imp)[-12:]
        ax.barh(np.array(features)[o], imp[o], color="#0b7285"); ax.set_title(f"RF importance: {t}")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=130); plt.close(fig)


def storm_maps(scores_meta, df, out, year_min, n=60):
    """Observed tracks of unseen storms, coloured by intensity, one panel per basin."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9)); axes = axes.ravel()
    d = df[df["year0"] >= year_min]
    for ax, (bi, b) in zip(axes, enumerate(BASINS)):
        s = d[d["basin"] == bi]
        ids = s["seg_id"].unique()[:n]
        s = s[s["seg_id"].isin(ids)]
        lo = s["lon"].values % 360                       # 0-360 keeps Pacific basins contiguous
        ax.scatter(lo, s["lat"], c=s["v"], s=4, cmap="turbo", vmin=15, vmax=150)
        ax.set_title(f"{BASIN_NAMES[b]}: unseen-storm tracks"); ax.grid(alpha=.3)
    axes[-1].axis("off")
    fig.tight_layout(); fig.savefig(out / "map_unseen_tracks_by_basin.png", dpi=110); plt.close(fig)
