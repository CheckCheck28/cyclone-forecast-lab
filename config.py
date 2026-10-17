"""Shared configuration for the global tropical-cyclone forecasting pipeline."""
from pathlib import Path

# --- Data -------------------------------------------------------------------
IBTRACS_URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-"
               "ibtracs/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv")
DATA_DIR = Path("data")
IBTRACS_CSV = DATA_DIR / "ibtracs.since1980.list.v04r01.csv"
OUT_DIR = Path("outputs")

# --- Geography --------------------------------------------------------------
BASINS = ["NA", "EP", "WP", "NI", "SI", "SP", "SA"]
BASIN_NAMES = {
    "NA": "North Atlantic", "EP": "East & Central Pacific", "WP": "Western Pacific",
    "NI": "North Indian", "SI": "South Indian", "SP": "South Pacific", "SA": "South Atlantic",
}
# Approximate map windows (lat_min, lat_max, lon_min, lon_max) used for plots
BASIN_BOX = {
    "NA": (0, 50, -100, 0), "EP": (0, 40, -180, -85), "WP": (0, 50, 100, 180),
    "NI": (0, 30, 40, 100), "SI": (-40, 0, 20, 135), "SP": (-40, 0, 135, 230), "SA": (-40, 0, -70, 20),
}

# --- Forecast set-up --------------------------------------------------------
STEP_H = 6                                  # best-track time step (hours)
LEADS = [6, 12, 18, 24, 36, 48, 60, 72]     # forecast lead times (hours)
HIST_STEPS = (1, 2, 4)                      # history lags used as features: 6h, 12h, 24h

# Chronological split by storm start year (no storm is ever split across sets)
TRAIN_END_YEAR = 2011      # train: <= 2011
VAL_END_YEAR = 2015        # validation (early stopping, ensemble weights): 2012-2015
                           # test / "unseen storms": >= 2016

# --- Model hyper-parameters -------------------------------------------------
GBM_PARAMS = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0)
RF_PARAMS = dict(n_estimators=50, max_depth=8, min_samples_leaf=60, max_features=0.6, max_samples=0.35)
KNN_PARAMS = dict(n_neighbors=30, weights="distance")
KNN_MAX_ROWS = 120_000
RANDOM_STATE = 42

KM_PER_DEG = 111.32
PRES_BAND_DEG = 5           # latitude band width of the intensity-climatology table
