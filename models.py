"""Model training: a core pair (gradient boosting + random forest, which also run inside the browser app)
plus a "model zoo" of ~15 other learners benchmarked on the same hold-out storms.

Modelling choices shared by every model
---------------------------------------
1. DIRECT multi-horizon forecasts (lead time is a feature) -> no compounding roll-out error.
2. Targets are km (east/north) residuals from constant-velocity extrapolation, and wind change in kt.
3. Hemisphere / basin / aligned-season features + a basin-month-latitude intensity-potential proxy.
4. Chronological storm-wise split (train <=2011, val 2012-15, test >=2016); validation years are used for
   early stopping and for the ensemble / stacking weights only.

Add your own model: write a function with the @zoo(...) decorator below returning an object with .predict(X) -> (n, 3).
"""
import importlib
import time
import numpy as np
from scipy.optimize import nnls
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from config import GBM_PARAMS, RF_PARAMS, KNN_PARAMS, KNN_MAX_ROWS, RANDOM_STATE
from features import FEATURES

TARGETS = ["dx", "dy", "dv"]          # residual east km, residual north km, wind change kt
RS = RANDOM_STATE
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
GBM_LABEL = "XGBoost" if HAS_XGB else "Gradient Boosting"
LABELS = {"gbm": GBM_LABEL, "rf": "Random Forest", "ens": "Ensemble (GBM+RF)",
          "stack": "Stacked ensemble (NNLS)", "base": "Persistence baseline"}


# ----------------------------------------------------------------------------- small wrappers
class PerTarget:
    """One 1-output model per target."""
    def __init__(self, models): self.models = models
    def predict(self, X): return np.column_stack([m.predict(X) for m in self.models])


class Scaled:
    """Standardise targets so km and kt are weighted equally by multi-output losses."""
    def __init__(self, model, mu, sd): self.model, self.mu, self.sd = model, mu, sd
    def predict(self, X): return self.model.predict(X) * self.sd + self.mu


class ClimMean:
    """Mean training residual per (basin, lead): a 'bias-corrected persistence' benchmark."""
    def fit(self, X, y):
        bcols = [FEATURES.index(f"b_{b}") for b in ("NA", "EP", "WP", "NI", "SI", "SP", "SA")]
        self.bcols, self.lc = bcols, FEATURES.index("lead")
        key = np.argmax(X[:, bcols], 1) * 1000 + X[:, self.lc].astype(int)
        self.table = {k: y[key == k].mean(0) for k in np.unique(key)}
        self.default = y.mean(0)
        return self
    def predict(self, X):
        key = np.argmax(X[:, self.bcols], 1) * 1000 + X[:, self.lc].astype(int)
        return np.array([self.table.get(k, self.default) for k in key])


# ----------------------------------------------------------------------------- the zoo
ZOO = {}


def zoo(key, label, cap=None, slow=False, needs=None):
    def deco(fn):
        ZOO[key] = dict(label=label, fn=fn, cap=cap, slow=slow, needs=needs)
        LABELS[key] = label
        return fn
    return deco


def _clip4(Z):
    return np.clip(Z, -4.0, 4.0)


def _ttr(reg):
    return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())


@zoo("climmean", "Climatological mean (basin, lead)")
def _climmean(Xt, yt, Xv, yv, fast): return ClimMean().fit(Xt, yt)

@zoo("ridge", "Ridge regression")
def _ridge(Xt, yt, Xv, yv, fast): return make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xt, yt)

@zoo("elasticnet", "Elastic Net")
def _enet(Xt, yt, Xv, yv, fast):
    return make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)).fit(Xt, yt)

@zoo("poly_ridge", "Polynomial (deg 2) Ridge", cap=120_000, slow=True)
def _poly(Xt, yt, Xv, yv, fast):
    clip = FunctionTransformer(_clip4)                              # stop quadratic terms exploding on outliers
    return make_pipeline(StandardScaler(), clip, PolynomialFeatures(2, include_bias=False), StandardScaler(),
                         Ridge(alpha=2000.0)).fit(Xt, yt)

@zoo("tree", "Decision tree")
def _tree(Xt, yt, Xv, yv, fast):
    return DecisionTreeRegressor(max_depth=10, min_samples_leaf=60, random_state=RS).fit(Xt, yt)

@zoo("extra", "Extra Trees", cap=300_000)
def _extra(Xt, yt, Xv, yv, fast):
    return ExtraTreesRegressor(n_estimators=12 if fast else 40, max_depth=10, min_samples_leaf=60, max_features=0.7,
                               n_jobs=-1, random_state=RS).fit(Xt, yt)

@zoo("hgb", "Hist Gradient Boosting (sklearn)")
def _hgb(Xt, yt, Xv, yv, fast):
    return PerTarget([HistGradientBoostingRegressor(max_iter=60 if fast else 300, learning_rate=0.06, max_depth=6,
                      l2_regularization=5.0, min_samples_leaf=60, random_state=RS).fit(Xt, yt[:, j]) for j in range(3)])

@zoo("gbr", "Gradient Boosting (sklearn, exact)", cap=150_000, slow=True)
def _gbr(Xt, yt, Xv, yv, fast):
    return PerTarget([GradientBoostingRegressor(n_estimators=40 if fast else 150, max_depth=4, learning_rate=0.08,
                      subsample=0.5, min_samples_leaf=40, random_state=RS).fit(Xt, yt[:, j]) for j in range(3)])

@zoo("ada", "AdaBoost (trees)", cap=100_000, slow=True)
def _ada(Xt, yt, Xv, yv, fast):
    return PerTarget([AdaBoostRegressor(DecisionTreeRegressor(max_depth=6, min_samples_leaf=50),
                      n_estimators=15 if fast else 60, learning_rate=0.5, random_state=RS).fit(Xt, yt[:, j]) for j in range(3)])

@zoo("knn", "KNN", cap=KNN_MAX_ROWS)
def _knn(Xt, yt, Xv, yv, fast):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_jobs=-1, **KNN_PARAMS)).fit(Xt, yt)

@zoo("mlp", "Neural net (MLP 128-64)", cap=300_000)
def _mlp(Xt, yt, Xv, yv, fast):
    reg = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, n_iter_no_change=6,
                        max_iter=12 if fast else 60, random_state=RS))
    return _ttr(reg).fit(Xt, yt)

@zoo("svr", "SVR (RBF kernel, subsampled)", cap=12_000, slow=True)
def _svr(Xt, yt, Xv, yv, fast):
    return PerTarget([_ttr(make_pipeline(StandardScaler(), SVR(C=3.0))).fit(Xt, yt[:, j]) for j in range(3)])

@zoo("lgbm", "LightGBM", needs="lightgbm")
def _lgbm(Xt, yt, Xv, yv, fast):
    import lightgbm as lgb
    ms = []
    for j in range(3):
        m = lgb.LGBMRegressor(n_estimators=100 if fast else 600, learning_rate=0.05, num_leaves=48, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=RS)
        ms.append(m.fit(Xt, yt[:, j], eval_set=[(Xv, yv[:, j])], callbacks=[lgb.early_stopping(40, verbose=False)]))
    return PerTarget(ms)

@zoo("catboost", "CatBoost (multi-target)", cap=400_000, needs="catboost")
def _cat(Xt, yt, Xv, yv, fast):
    from catboost import CatBoostRegressor
    mu, sd = yt.mean(0), yt.std(0) + 1e-6
    m = CatBoostRegressor(loss_function="MultiRMSE", iterations=100 if fast else 600, depth=6, learning_rate=0.08,
                          verbose=0, random_seed=RS)
    m.fit(Xt, (yt - mu) / sd, eval_set=(Xv, (yv - mu) / sd), early_stopping_rounds=40)
    return Scaled(m, mu, sd)


def available_models():
    out = []
    for k, s in ZOO.items():
        if s["needs"] is None or importlib.util.find_spec(s["needs"]) is not None:
            out.append(k)
    return out


def resolve_models(arg):
    """'all' | 'fast' (skips slow ones) | 'none' | comma-separated keys."""
    avail = available_models()
    if arg == "all":
        return avail
    if arg == "fast":
        return [k for k in avail if not ZOO[k]["slow"]]
    if arg == "none":
        return []
    keys = [k.strip() for k in arg.split(",") if k.strip()]
    bad = [k for k in keys if k not in avail]
    if bad:
        print(f"  (ignoring unknown/unavailable models: {bad}; available: {avail})")
    return [k for k in keys if k in avail]


# ----------------------------------------------------------------------------- training
def train_all(train, val, fast=False, zoo_keys=(), verbose=True):
    Xtr, ytr, Xva, yva = train["X"], train["y"], val["X"], val["y"]
    models, secs = {"gbm": [], "rf": [], "zoo": {}}, {}
    t0 = time.time()
    for j, t in enumerate(TARGETS):
        if verbose:
            print(f"  core [{t}] {GBM_LABEL} + Random Forest ...", flush=True)
        if HAS_XGB:
            p = dict(GBM_PARAMS, base_score=0.0, tree_method="hist", n_jobs=-1, random_state=RS, early_stopping_rounds=40)
            if fast:
                p["n_estimators"] = 80
            m = XGBRegressor(**p).fit(Xtr, ytr[:, j], eval_set=[(Xva, yva[:, j])], verbose=False)
        else:
            m = GradientBoostingRegressor(n_estimators=60 if fast else 150, max_depth=4, learning_rate=0.08,
                                          subsample=0.5, min_samples_leaf=40, random_state=RS).fit(Xtr, ytr[:, j])
        models["gbm"].append(m)
        rp = dict(RF_PARAMS, n_jobs=-1, random_state=RS)
        if fast:
            rp["n_estimators"] = 15
        models["rf"].append(RandomForestRegressor(**rp).fit(Xtr, ytr[:, j]))
    secs["gbm"] = secs["rf"] = round((time.time() - t0) / 2, 1)
    models["ens_w"] = ensemble_weights(models, Xva, yva)

    rng = np.random.default_rng(RS)
    for k in zoo_keys:
        s = ZOO[k]
        cap = s["cap"] and max(5000, s["cap"] // 4 if fast else s["cap"])
        idx = np.arange(len(Xtr)) if not cap or cap >= len(Xtr) else rng.choice(len(Xtr), cap, replace=False)
        if verbose:
            print(f"  zoo  {s['label']} (n={len(idx):,}) ...", flush=True)
        t0 = time.time()
        try:
            models["zoo"][k] = s["fn"](Xtr[idx], ytr[idx], Xva, yva, fast)
            secs[k] = round(time.time() - t0, 1)
        except Exception as e:                      # one failing model must not abort the run
            print(f"    skipped {k}: {type(e).__name__}: {e}")
    models["secs"] = secs
    models["stack_w"] = fit_stack(models, Xva, yva, verbose)
    models["order"] = model_order(models)
    return models


def model_order(models):
    return ["ens"] + (["stack"] if models.get("stack_w") else []) + ["gbm", "rf"] + list(models["zoo"]) + ["base"]


def _components(models, X):
    out = {"gbm": np.column_stack([m.predict(X) for m in models["gbm"]]),
           "rf": np.column_stack([m.predict(X) for m in models["rf"]])}
    for k, m in models["zoo"].items():
        out[k] = m.predict(X)
    return out


def predict_raw(models, X):
    """{model: (n,3) residual predictions} for every fitted model, the ensemble and the stack."""
    out = _components(models, X)
    w = np.asarray(models["ens_w"])
    out["ens"] = out["gbm"] * w[:, 0] + out["rf"] * w[:, 1]
    sw = models.get("stack_w")
    if sw:
        out["stack"] = sum(out[k] * np.asarray(v) for k, v in sw.items())
    return out


def ensemble_weights(models, Xva, yva):
    """Inverse-MSE weights per target for [gbm, rf] on the validation years (used by the browser app)."""
    g = np.column_stack([m.predict(Xva) for m in models["gbm"]])
    r = np.column_stack([m.predict(Xva) for m in models["rf"]])
    w = []
    for j in range(3):
        a, b = 1.0 / np.mean((g[:, j] - yva[:, j]) ** 2), 1.0 / np.mean((r[:, j] - yva[:, j]) ** 2)
        w.append([a / (a + b), b / (a + b)])
    return np.round(w, 4).tolist()


def fit_stack(models, Xva, yva, verbose=True):
    """Non-negative least-squares blend of ALL fitted models, learned on the validation years only."""
    if len(models["zoo"]) == 0:
        return None
    P = _components(models, Xva)
    keys = list(P)
    W = np.zeros((len(keys), 3))
    for j in range(3):
        W[:, j], _ = nnls(np.column_stack([P[k][:, j] for k in keys]), yva[:, j])
    sw = {k: np.round(W[i], 3).tolist() for i, k in enumerate(keys) if W[i].max() > 1e-3}
    if verbose:
        print("  stack weights (dx,dy,dv):", {k: v for k, v in sw.items()})
    return sw
