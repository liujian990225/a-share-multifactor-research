from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _allocate_industry_quota(group: pd.DataFrame, top_n: int, industry_col: str) -> dict[str, int]:
    counts = group[industry_col].fillna("Unknown").value_counts()
    raw = counts / counts.sum() * top_n
    quota = raw.apply(math.floor).astype(int)
    quota[quota == 0] = 1
    while quota.sum() > top_n:
        idx = quota[quota > 1].idxmax()
        quota.loc[idx] -= 1
    while quota.sum() < top_n:
        fractional = (raw - raw.apply(math.floor)).sort_values(ascending=False)
        for idx in fractional.index:
            quota.loc[idx] += 1
            if quota.sum() >= top_n:
                break
    return quota.to_dict()


def select_portfolio(
    signal_df: pd.DataFrame,
    score_col: str,
    top_n: int | None = 30,
    top_pct: float | None = None,
    industry_neutral: bool = True,
    industry_col: str = "industry",
) -> pd.DataFrame:
    """Select top stocks and assign equal weights."""
    group = signal_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col]).copy()
    if group.empty:
        return pd.DataFrame(columns=["date", "symbol", "weight", score_col, industry_col])

    if top_pct is not None:
        n_select = max(1, int(len(group) * top_pct))
    else:
        n_select = int(top_n or 30)
    n_select = min(n_select, len(group))

    if not industry_neutral or industry_col not in group.columns:
        selected = group.sort_values(score_col, ascending=False).head(n_select).copy()
    else:
        selected_parts = []
        quota = _allocate_industry_quota(group, n_select, industry_col)
        for industry, k in quota.items():
            sub = group[group[industry_col].fillna("Unknown") == industry]
            selected_parts.append(sub.sort_values(score_col, ascending=False).head(k))
        selected = pd.concat(selected_parts).sort_values(score_col, ascending=False).head(n_select).copy()
        if len(selected) < n_select:
            extra = group[~group["symbol"].isin(selected["symbol"])].sort_values(score_col, ascending=False).head(n_select - len(selected))
            selected = pd.concat([selected, extra], ignore_index=True)

    selected["weight"] = 1.0 / len(selected)
    keep_cols = ["date", "symbol", "weight", score_col]
    if industry_col in selected.columns:
        keep_cols.append(industry_col)
    if "size_log_mcap" in selected.columns:
        keep_cols.append("size_log_mcap")
    return selected[keep_cols].sort_values(["date", "weight", "symbol"], ascending=[True, False, True]).reset_index(drop=True)


def build_target_weights(
    panel: pd.DataFrame,
    score_col: str,
    top_n: int | None = 30,
    top_pct: float | None = None,
    industry_neutral: bool = True,
) -> pd.DataFrame:
    """Build target portfolio weights for every rebalance date."""
    parts = []
    for _, group in panel.groupby("date", sort=True):
        selected = select_portfolio(
            group,
            score_col=score_col,
            top_n=top_n,
            top_pct=top_pct,
            industry_neutral=industry_neutral,
        )
        if not selected.empty:
            parts.append(selected)
    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "weight", score_col, "industry"])
    return pd.concat(parts, ignore_index=True)


def calculate_industry_exposure(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty or "industry" not in weights.columns:
        return pd.DataFrame()
    exposure = weights.groupby(["date", "industry"], as_index=False)["weight"].sum()
    exposure = exposure.rename(columns={"weight": "industry_weight"})
    return exposure


def calculate_style_exposure(weights: pd.DataFrame, panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """Weighted average portfolio exposure to selected style factors."""
    if weights.empty:
        return pd.DataFrame()
    style_cols = [col for col in factor_cols + ["size_log_mcap"] if col in panel.columns]
    merged = weights[["date", "symbol", "weight"]].merge(panel[["date", "symbol"] + style_cols], on=["date", "symbol"], how="left")
    records = []
    for date, group in merged.groupby("date"):
        row = {"date": date}
        for col in style_cols:
            row[col] = (group[col] * group["weight"]).sum(skipna=True)
        records.append(row)
    return pd.DataFrame(records).sort_values("date")
