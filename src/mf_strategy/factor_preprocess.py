from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def _fill_missing_by_industry(group: pd.DataFrame, factor_cols: list[str], industry_col: str) -> pd.DataFrame:
    out = group.copy()
    for col in factor_cols:
        if industry_col in out.columns:
            out[col] = out[col].fillna(out.groupby(industry_col)[col].transform("median"))
        out[col] = out[col].fillna(out[col].median())
    return out


def _mad_clip(series: pd.Series, n: float = 3.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    median = s.median()
    mad = (s - median).abs().median()
    if pd.isna(median) or pd.isna(mad) or mad == 0:
        lower, upper = s.quantile([0.01, 0.99])
    else:
        scaled_mad = 1.4826 * mad
        lower, upper = median - n * scaled_mad, median + n * scaled_mad
    return s.clip(lower, upper)


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _neutralize_group(
    group: pd.DataFrame,
    factor_cols: list[str],
    industry_col: str = "industry",
    size_col: str = "size_log_mcap",
    neutralize_size: bool = True,
) -> pd.DataFrame:
    out = group.copy()
    if industry_col not in out.columns:
        return out

    x_parts = [pd.Series(1.0, index=out.index, name="const")]
    if neutralize_size and size_col in out.columns:
        x_parts.append(pd.to_numeric(out[size_col], errors="coerce").fillna(out[size_col].median()).rename(size_col))
    dummies = pd.get_dummies(out[industry_col].fillna("Unknown"), prefix="ind", drop_first=True, dtype=float)
    if not dummies.empty:
        x_parts.append(dummies)
    X = pd.concat(x_parts, axis=1).astype(float)

    for col in factor_cols:
        y = pd.to_numeric(out[col], errors="coerce")
        mask = y.notna() & np.isfinite(y)
        if mask.sum() <= X.shape[1] + 5:
            continue
        X_masked = X.loc[mask].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_masked = y.loc[mask]
        try:
            beta, *_ = np.linalg.lstsq(X_masked.to_numpy(), y_masked.to_numpy(), rcond=None)
            fitted = X_masked.to_numpy() @ beta
            residual = y_masked.to_numpy() - fitted
            out.loc[mask, col] = residual
        except np.linalg.LinAlgError:
            continue
    return out


def preprocess_factor_panel(
    panel: pd.DataFrame,
    factor_cols: list[str],
    config: Mapping[str, Any],
    industry_col: str = "industry",
) -> pd.DataFrame:
    """Cross-sectionally clean, clip, standardize and neutralize factors by date."""
    factor_cfg = config.get("factor", {})
    mad_n = float(factor_cfg.get("mad_n", 3.0))
    do_neutralize = bool(factor_cfg.get("neutralize", True))
    neutralize_size = bool(factor_cfg.get("neutralize_size", True))

    processed_parts = []
    for date, group in panel.groupby("date", sort=True):
        g = _fill_missing_by_industry(group, factor_cols, industry_col)
        for col in factor_cols:
            g[col] = _mad_clip(g[col], n=mad_n)
            g[col] = _zscore(g[col])
        if do_neutralize:
            g = _neutralize_group(g, factor_cols, industry_col=industry_col, neutralize_size=neutralize_size)
            for col in factor_cols:
                g[col] = _zscore(g[col])
        g["date"] = date
        processed_parts.append(g)

    return pd.concat(processed_parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def add_category_scores(
    panel: pd.DataFrame,
    factor_categories: dict[str, list[str]],
    category_weights: Mapping[str, float] | None = None,
    score_col: str = "score_equal",
) -> pd.DataFrame:
    """Add category-level scores and an equal/static weighted composite score."""
    out = panel.copy()
    if category_weights is None:
        category_weights = {cat: 1 / len(factor_categories) for cat in factor_categories}

    weight_sum = sum(float(v) for v in category_weights.values()) or 1.0
    for category, cols in factor_categories.items():
        valid_cols = [col for col in cols if col in out.columns]
        out[f"score_{category}"] = out[valid_cols].mean(axis=1) if valid_cols else 0.0

    out[score_col] = 0.0
    for category in factor_categories:
        out[score_col] += float(category_weights.get(category, 0.0)) / weight_sum * out[f"score_{category}"]
    return out


def add_rolling_ic_weighted_score(
    panel: pd.DataFrame,
    factor_cols: list[str],
    score_col: str = "score_ic_weighted",
    lookback: int = 12,
    min_obs: int = 6,
    fwd_return_col: str = "fwd_return",
) -> pd.DataFrame:
    """Create a dynamic score using only past Rank IC observations.

    For each rebalance date, factor weights are proportional to positive shifted rolling
    mean Rank IC. If there are too few observations or all means are non-positive, it
    falls back to equal weights.
    """
    out = panel.copy()
    dates = sorted(out["date"].dropna().unique())
    ic_history: dict[pd.Timestamp, dict[str, float]] = {}

    for date, group in out.groupby("date", sort=True):
        current: dict[str, float] = {}
        valid_y = group[fwd_return_col]
        for col in factor_cols:
            if col not in group.columns:
                continue
            valid = group[[col, fwd_return_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 10 or valid[col].nunique() <= 1 or valid[fwd_return_col].nunique() <= 1:
                current[col] = np.nan
            else:
                current[col] = valid[col].rank().corr(valid[fwd_return_col].rank())
        ic_history[pd.Timestamp(date)] = current

    ic_df = pd.DataFrame.from_dict(ic_history, orient="index").sort_index()
    rolling_ic = ic_df.shift(1).rolling(window=lookback, min_periods=min_obs).mean()

    out[score_col] = np.nan
    equal_weights = pd.Series(1.0 / len(factor_cols), index=factor_cols)
    for date in dates:
        date_ts = pd.Timestamp(date)
        weights = rolling_ic.loc[date_ts] if date_ts in rolling_ic.index else equal_weights
        weights = weights.reindex(factor_cols).fillna(0.0)
        weights = weights.clip(lower=0.0)
        if weights.sum() <= 0:
            weights = equal_weights
        else:
            weights = weights / weights.sum()
        mask = out["date"] == date_ts
        score = out.loc[mask, factor_cols].mul(weights, axis=1).sum(axis=1)
        out.loc[mask, score_col] = score.values

    out[score_col] = out.groupby("date")[score_col].transform(_zscore)
    return out
