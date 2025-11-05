from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from scipy import stats
from scipy.special import inv_boxcox

import matplotlib.pyplot as plt

try:
    import pingouin as pg  # optional
except Exception:
    pg = None

__all__ = [
    "transform_series",
    "invert_series",
    "transform_df",
    "invert_df",
    "normality_report",
    "plot_hist",
]

# -----------------------------
# Internal helpers
# -----------------------------

def _handle_nans(x: pd.Series, strategy: str):
    """Handle NaNs according to the chosen strategy: 'ignore' | 'drop' | 'impute_mean'."""
    meta = {}
    if strategy == "drop":
        mask = x.isna()
        meta["nan_mask"] = mask.to_list()
        return x[~mask], meta
    elif strategy == "impute_mean":
        m = x.mean()
        meta["impute_mean"] = None if pd.isna(m) else float(m)
        return x.fillna(m), meta
    else:  # ignore
        return x, meta

def _restore_nans(y: pd.Series, meta: Dict[str, Any]) -> pd.Series:
    """Reinsert NaNs for the 'drop' strategy after transformation."""
    if "nan_mask" in meta:
        import numpy as np
        mask = np.array(meta["nan_mask"], dtype=bool)
        out = pd.Series(index=np.arange(mask.shape[0]), dtype=float)
        out[mask] = np.nan
        out[~mask] = y.values
        out.index = y.index.union(out.index)
        out = out.loc[y.index]
        return out
    return y

def _inverse_yeojohnson(y: np.ndarray, lam: float) -> np.ndarray:
    """Inverse of the Yeo-Johnson transform (see SciPy docs)."""
    out = np.empty_like(y, dtype=float)
    pos = y >= 0
    if lam != 0:
        out[pos] = np.power(y[pos] * lam + 1.0, 1.0 / lam) - 1.0
    else:
        out[pos] = np.exp(y[pos]) - 1.0
    if lam != 2:
        out[~pos] = 1.0 - np.power(1.0 - (2.0 - lam) * y[~pos], 1.0 / (2.0 - lam))
    else:
        out[~pos] = 1.0 - np.exp(-y[~pos])
    return out

# -----------------------------
# Public API
# -----------------------------

def transform_series(x: pd.Series, method: str, standardize: bool = True, na_action: str = "ignore") -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Transform a single pandas Series using 'log', 'boxcox', or 'yeojohnson'.
    Optionally standardize (z-score). Returns (transformed_series, params_dict).
    """
    params: Dict[str, Any] = {"method": method}
    x_in, na_meta = _handle_nans(x, na_action)
    params.update({"na_meta": na_meta, "na_action": na_action})

    # Prepare positivity / shifting if needed
    if method in ("log", "boxcox"):
        min_x = x_in.min(skipna=True)
        shift = 0.0
        if pd.notna(min_x) and min_x <= 0:
            shift = float(1e-6 - min_x)
        params["shift"] = shift
        x_pos = x_in + shift
        eps = float(1e-6)
        x_pos = x_pos.clip(lower=eps)
    elif method == "yeojohnson":
        params["shift"] = 0.0
        x_pos = x_in
    else:
        raise ValueError("method must be one of {'log','boxcox','yeojohnson'}")

    # Core transform
    if method == "log":
        y_core = np.log(x_pos.astype(float))
        lam = None
    elif method == "boxcox":
        y_core, lam = stats.boxcox(x_pos.astype(float))
    else:  # yeojohnson
        y_core, lam = stats.yeojohnson(x_pos.astype(float))
    params["lambda"] = None if lam is None else float(lam)

    y = pd.Series(y_core, index=x_in.index, dtype=float)

    # Standardize
    if standardize:
        mu = float(y.mean())
        sigma = float(y.std(ddof=0)) or 1.0
        y = (y - mu) / sigma
        params.update({"standardize": True, "mean": mu, "std": sigma})
    else:
        params.update({"standardize": False, "mean": None, "std": None})

    # Reinsert NaNs according to strategy
    if na_action == "drop":
        y_full = _restore_nans(y, na_meta)
    elif na_action == "ignore":
        y_full = x.copy().astype(float)
        y_full.loc[y.index] = y.values
    else:  # impute_mean
        y_full = y.reindex_like(x)
    return y_full, params

def invert_series(y: pd.Series, params: Dict[str, Any]) -> pd.Series:
    """
    Invert a transformed Series using the parameter dict returned by transform_series/transform_df.
    """
    method = params.get("method")
    lam = params.get("lambda", None)
    shift = float(params.get("shift", 0.0))
    standardize = bool(params.get("standardize", False))

    x = y.astype(float).copy()
    if standardize:
        mu = float(params.get("mean", 0.0))
        sigma = float(params.get("std", 1.0)) or 1.0
        x = x * sigma + mu

    if method == "log":
        x = np.exp(x) - shift
    elif method == "boxcox":
        if lam is None:
            raise ValueError("Missing lambda for Box-Cox inverse.")
        x = inv_boxcox(x, lam) - shift
    elif method == "yeojohnson":
        if lam is None:
            raise ValueError("Missing lambda for Yeo-Johnson inverse.")
        x = _inverse_yeojohnson(x.values, lam)
    else:
        raise ValueError("Unknown method for inversion.")
    return pd.Series(x, index=y.index)

def transform_df(df: pd.DataFrame, columns: List[str] = None, method: str = "yeojohnson", standardize: bool = True, na_action: str = "ignore"):
    """
    Transform a DataFrame across selected columns. If columns=None, all numeric columns are used.
    Returns (df_transformed, params_all), where params_all contains per-column parameters.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    params_all: Dict[str, Any] = {"method": method, "standardize": standardize, "na_action": na_action, "columns": {}}
    out = df.copy()
    for c in columns:
        y, p = transform_series(out[c], method, standardize, na_action)
        out[c] = y
        params_all["columns"][c] = p
    return out, params_all

def invert_df(df: pd.DataFrame, params_all: Dict[str, Any]):
    """
    Invert a previously transformed DataFrame using the params_all dict returned by transform_df.
    """
    out = df.copy()
    for c, p in params_all.get("columns", {}).items():
        if c in out.columns:
            out[c] = invert_series(out[c], p)
    return out

def normality_report(df_before: pd.DataFrame, df_after: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Quick normality summary: skewness and Shapiro-Wilk p-values before/after.
    If pingouin is available, uses pg.normality; otherwise falls back to scipy.stats.shapiro.
    """
    rows = []
    for c in cols:
        s0 = df_before[c].dropna().values
        s1 = df_after[c].dropna().values

        skew0 = stats.skew(s0) if s0.size > 2 else np.nan
        skew1 = stats.skew(s1) if s1.size > 2 else np.nan

        if pg is not None:
            sw0 = pg.normality(df_before[[c]].dropna(), method="shapiro")["pval"].values[0]
            sw1 = pg.normality(df_after[[c]].dropna(), method="shapiro")["pval"].values[0]
        else:
            sw0 = stats.shapiro(s0)[1] if 3 <= s0.size <= 5000 else np.nan
            sw1 = stats.shapiro(s1)[1] if 3 <= s1.size <= 5000 else np.nan

        rows.append({
            "column": c,
            "skew_before": skew0,
            "skew_after": skew1,
            "shapiro_p_before": sw0,
            "shapiro_p_after": sw1,
        })
    return pd.DataFrame(rows)

def plot_hist(series: pd.Series, title: str):
    """
    Simple helper to plot a histogram for a Series (no specific styles/colors).
    """
    plt.figure()
    plt.hist(series.dropna().values, bins=40)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.show()
