"""Serialise trained models + statistics + a library of unseen storms into one JSON bundle and inline it
into the HTML app, so the browser runs the SAME trees as Python (no server needed)."""
import json
import numpy as np
from pathlib import Path
from config import LEADS, BASINS, BASIN_NAMES, STEP_H
from features import FEATURES, compute_features, km_to_latlon
from models import HAS_XGB, GBM_LABEL, predict_raw


# ----------------------------------------------------------------------------- tree flattening
class Flat:
    """Concatenated trees: pred = base + sum(leaf values along each tree). Leaves have f = -1."""
    def __init__(self, lt):
        self.f, self.t, self.l, self.r, self.v, self.o, self.lt = [], [], [], [], [], [], lt

    def add(self, feat, thr, left, right, val):
        off = len(self.f)
        self.o.append(off)
        self.f += [int(x) for x in feat]
        self.t += [round(float(x), 7) for x in thr]
        self.l += [(off + int(x)) if x >= 0 else -1 for x in left]
        self.r += [(off + int(x)) if x >= 0 else -1 for x in right]
        self.v += [round(float(x), 4) for x in val]

    def dump(self, base):
        return dict(base=round(float(base), 4), lt=self.lt, f=self.f, t=self.t, l=self.l, r=self.r, v=self.v, o=self.o)


def _xgb_tree(node, feat, thr, left, right, val):
    """Depth-first flatten of an XGBoost JSON dump node; returns its index."""
    i = len(feat)
    feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); val.append(0.0)
    if "leaf" in node:
        val[i] = node["leaf"]
        return i
    kids = {c["nodeid"]: c for c in node["children"]}
    name = node["split"]
    feat[i] = FEATURES.index(name) if name in FEATURES else int(name.lstrip("f"))
    thr[i] = node["split_condition"]
    left[i] = _xgb_tree(kids[node["yes"]], feat, thr, left, right, val)
    right[i] = _xgb_tree(kids[node["no"]], feat, thr, left, right, val)
    return i


def export_gbm(model):
    if HAS_XGB and hasattr(model, "get_booster"):
        bst = model.get_booster()
        if getattr(model, "best_iteration", None) is not None:
            bst = bst[: model.best_iteration + 1]
        flat = Flat(lt=True)                                   # xgboost: go left if x < threshold
        for s in bst.get_dump(dump_format="json"):
            f, t, l, r, v = [], [], [], [], []
            _xgb_tree(json.loads(s), f, t, l, r, v)
            flat.add(f, t, l, r, v)
        return flat.dump(base=0.0)                             # trained with base_score=0
    flat = Flat(lt=False)                                      # sklearn: go left if x <= threshold
    lr = model.learning_rate
    for est in model.estimators_[:, 0]:
        tr = est.tree_
        leaf = tr.children_left < 0
        flat.add(np.where(leaf, -1, tr.feature), tr.threshold, tr.children_left, tr.children_right,
                 np.where(leaf, tr.value[:, 0, 0] * lr, 0.0))
    return flat.dump(base=float(model.init_.constant_.ravel()[0]))


def export_rf(model):
    flat = Flat(lt=False)
    n = len(model.estimators_)
    for est in model.estimators_:
        tr = est.tree_
        leaf = tr.children_left < 0
        flat.add(np.where(leaf, -1, tr.feature), tr.threshold, tr.children_left, tr.children_right,
                 np.where(leaf, tr.value[:, 0, 0] / n, 0.0))
    return flat.dump(base=0.0)


# ----------------------------------------------------------------------------- storm library
def storm_library(df, min_year):
    """Tracks of storms from the unseen (test) years, split into gap-free segments."""
    lib = []
    d = df[df["year0"] >= min_year]
    for sid, g in d.groupby("seg_id", sort=False):
        if len(g) < 10:
            continue
        g = g.sort_values("time")
        lib.append(dict(id=sid, n=g["name"].iloc[0].title(), y=int(g["year0"].iloc[0]),
                        b=BASINS[int(g["basin"].iloc[0])],
                        t0=int(g["time"].iloc[0].timestamp() // 3600),
                        p=[[round(a, 1), round(b, 1), int(c)] for a, b, c in zip(g["lat"], g["lon"], g["v"])]))
    lib.sort(key=lambda s: (-s["y"], s["b"], s["n"]))
    return lib


NAMES = ("dx", "dy", "dv")


def _r(a, d=4):
    return np.round(np.asarray(a, float), d).tolist()


def _tree_set(trees, scale=1.0):
    """Multi-output sklearn trees -> one flat tree-set per target."""
    out = {}
    for j, n in enumerate(NAMES):
        flat = Flat(lt=False)
        for tr in trees:
            t = tr.tree_
            leaf = t.children_left < 0
            flat.add(np.where(leaf, -1, t.feature), t.threshold, t.children_left, t.children_right,
                     np.where(leaf, t.value[:, j, 0] * scale, 0.0))
        out[n] = flat.dump(0.0)
    return dict(t="trees", m=out)


def _hgb(m):
    flat = Flat(lt=False)
    for it in m._predictors:
        nd = it[0].nodes
        leaf = nd["is_leaf"].astype(bool)
        flat.add(np.where(leaf, -1, nd["feature_idx"]), nd["num_threshold"], np.where(leaf, -1, nd["left"]),
                 np.where(leaf, -1, nd["right"]), np.where(leaf, nd["value"], 0.0))
    return flat.dump(base=float(np.ravel(m._baseline_prediction)[0]))


def _lin(pipe):
    sc, lm = pipe.steps[0][1], pipe.steps[-1][1]
    return dict(t="lin", mu=_r(sc.mean_, 9), sd=_r(sc.scale_, 9), w=_r(lm.coef_, 5), b=_r(lm.intercept_, 5))


def _mlp(ttr):
    pipe = ttr.regressor_
    sc, mlp, ys = pipe.steps[0][1], pipe.steps[-1][1], ttr.transformer_
    return dict(t="mlp", mu=_r(sc.mean_, 9), sd=_r(sc.scale_, 9), W=[_r(w.T, 6) for w in mlp.coefs_],
                b=[_r(b, 6) for b in mlp.intercepts_], ymu=_r(ys.mean_, 6), ysd=_r(ys.scale_, 6))


def _tab(m):
    return dict(t="tab", k={str(k): _r(v, 3) for k, v in m.table.items()}, d=_r(m.default, 3))


def export_zoo(models):
    """Every benchmark model that can be evaluated in plain JavaScript. Others stay Python-only."""
    out = {}
    for k, m in models["zoo"].items():
        try:
            if k == "extra":
                out[k] = _tree_set(m.estimators_, 1.0 / len(m.estimators_))
            elif k == "tree":
                out[k] = _tree_set([m])
            elif k == "gbr":
                out[k] = dict(t="trees", m={n: export_gbm(e) for n, e in zip(NAMES, m.models)})
            elif k == "hgb":
                out[k] = dict(t="trees", m={n: _hgb(e) for n, e in zip(NAMES, m.models)})
            elif k in ("ridge", "elasticnet"):
                out[k] = _lin(m)
            elif k == "mlp":
                out[k] = _mlp(m)
            elif k == "climmean":
                out[k] = _tab(m)
        except Exception as e:                                   # never let an exporter break the build
            print(f"  (not exported to the app: {k}: {type(e).__name__}: {e})")
    return out


def selftest_vectors(models, clim, browser, n=12, seed=3):
    """Random states + Python's residual predictions from EVERY browser model; the app re-computes
    them in JavaScript at load time and logs the maximum deviation."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        lat, lon = rng.uniform(5, 30) * rng.choice([-1, 1]), rng.uniform(-180, 180)
        v, dv = rng.uniform(25, 120), rng.uniform(-20, 25)
        e, nn = rng.uniform(-25, 10), rng.uniform(-5, 15)
        m = int(rng.integers(1, 13)); doy = int((m - 1) * 30.4 + 15)
        mk = lambda k: ((lat - nn * k), (lon - e * k / (111.32 * max(np.cos(np.radians(lat)), .05))), v - dv * k / 24)
        (a6, b6, c6), (a12, b12, c12), (a24, b24, c24) = mk(6), mk(12), mk(24)
        Xb, _ = compute_features(lat, lon, v, a6, b6, c6, a12, b12, c12, a24, b24, c24, m, doy, clim)
        rows = np.column_stack([np.repeat(Xb, len(LEADS), 0), np.array(LEADS)]).astype(np.float32)
        raw = predict_raw(models, rows)
        out.append(dict(inp=[lat, lon, v, a6, b6, c6, a12, b12, c12, a24, b24, c24, m, doy],
                        P={k: [raw[k][:, j].round(3).tolist() for j in range(3)] for k in browser if k in raw}))
    return out


def build_bundle(models, clim, pres, stats, lib, activity, info):
    zoo = export_zoo(models)
    sw = models.get("stack_w")
    stack_ok = bool(sw) and all(k in zoo or k in ("gbm", "rf") for k in sw)
    browser = ["ens"] + (["stack"] if stack_ok else []) + ["gbm", "rf"] + list(zoo) + ["base"]
    return dict(
        info=info, features=FEATURES, leads=LEADS, basins=BASINS, basin_names=BASIN_NAMES, step_h=STEP_H,
        clim=np.asarray(clim).tolist(), pres=pres, ens_w=models["ens_w"], gbm_label=GBM_LABEL,
        trees={"gbm": {n: export_gbm(m) for n, m in zip(NAMES, models["gbm"])},
               "rf": {n: export_rf(m) for n, m in zip(NAMES, models["rf"])}},
        zoo=zoo, stack_w=sw if stack_ok else None, browser=browser,
        stats=stats, storms=lib, activity=activity,
        selftest=selftest_vectors(models, clim, [b for b in browser if b != "base"]))


def write_app(bundle, template_path, out_html, out_json):
    """Write the JSON bundle and a single self-contained HTML (data inlined)."""
    blob = json.dumps(bundle, separators=(",", ":"), allow_nan=False)
    Path(out_json).write_text(blob)
    html = Path(template_path).read_text(encoding="utf-8")
    Path(out_html).write_text(html.replace("/*__APP_DATA__*/null", blob), encoding="utf-8")
    print(f"App bundle: {len(blob)/1e6:.1f} MB -> {out_html}")
