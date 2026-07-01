from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward_returns(panel: pd.DataFrame, prices: pd.DataFrame, rebalance_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Add next-period close-to-close forward returns for factor testing."""
    out = panel.copy()
    pivot = prices.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    rebal_dates = [pd.Timestamp(d) for d in rebalance_dates if pd.Timestamp(d) in pivot.index]
    ret_rows = []
    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]
        current = pivot.loc[date]
        nxt = pivot.loc[next_date]
        fwd = nxt / current - 1.0
        ret_rows.append(pd.DataFrame({"date": date, "symbol": fwd.index, "fwd_return": fwd.values}))
    if not ret_rows:
        out["fwd_return"] = np.nan
        return out
    fwd_df = pd.concat(ret_rows, ignore_index=True)
    out = out.merge(fwd_df, on=["date", "symbol"], how="left")
    return out


def calculate_ic(panel: pd.DataFrame, factor_cols: list[str], fwd_return_col: str = "fwd_return") -> pd.DataFrame:
    """Calculate Pearson IC and Spearman Rank IC for each factor by date."""
    records = []
    for date, group in panel.groupby("date", sort=True):
        for factor in factor_cols:
            valid = group[[factor, fwd_return_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 10 or valid[factor].nunique() <= 1 or valid[fwd_return_col].nunique() <= 1:
                pearson = np.nan
                rank_ic = np.nan
            else:
                pearson = valid[factor].corr(valid[fwd_return_col], method="pearson")
                rank_ic = valid[factor].corr(valid[fwd_return_col], method="spearman")
            records.append({"date": date, "factor": factor, "ic": pearson, "rank_ic": rank_ic, "n": len(valid)})
    return pd.DataFrame(records)


def summarize_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize IC statistics by factor."""
    rows = []
    for factor, group in ic_df.groupby("factor"):
        for col in ["ic", "rank_ic"]:
            s = group[col].dropna()
            if s.empty:
                mean = std = icir = win_rate = np.nan
            else:
                mean = s.mean()
                std = s.std(ddof=1)
                icir = mean / std if std and not np.isnan(std) else np.nan
                win_rate = (s > 0).mean()
            rows.append(
                {
                    "factor": factor,
                    "metric": col,
                    "mean": mean,
                    "std": std,
                    "icir": icir,
                    "win_rate": win_rate,
                    "obs": int(s.shape[0]),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "icir"], ascending=[True, False])


def quantile_analysis(
    panel: pd.DataFrame,
    score_col: str,
    n_quantiles: int = 5,
    fwd_return_col: str = "fwd_return",
) -> pd.DataFrame:
    """Calculate forward return by score quantile for every rebalance date."""
    records = []
    for date, group in panel.groupby("date", sort=True):
        valid = group[[score_col, fwd_return_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < n_quantiles * 3 or valid[score_col].nunique() < n_quantiles:
            continue
        try:
            labels = [f"Q{i + 1}" for i in range(n_quantiles)]
            valid = valid.copy()
            valid["quantile"] = pd.qcut(valid[score_col], q=n_quantiles, labels=labels, duplicates="drop")
            for quantile, q_group in valid.groupby("quantile", observed=False):
                records.append(
                    {
                        "date": date,
                        "score": score_col,
                        "quantile": str(quantile),
                        "mean_fwd_return": q_group[fwd_return_col].mean(),
                        "count": len(q_group),
                    }
                )
        except ValueError:
            continue
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result = result.sort_values(["score", "quantile", "date"])
    result["cum_return"] = result.groupby(["score", "quantile"])["mean_fwd_return"].transform(lambda s: (1 + s).cumprod() - 1)
    return result


def factor_correlation(panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """Average cross-sectional factor correlation matrix across rebalance dates."""
    matrices = []
    for _, group in panel.groupby("date"):
        corr = group[factor_cols].corr()
        matrices.append(corr)
    if not matrices:
        return pd.DataFrame(index=factor_cols, columns=factor_cols)
    return sum(matrices) / len(matrices)


def rolling_factor_decay(ic_df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Calculate rolling Rank IC mean as a simple factor decay/stability diagnostic."""
    parts = []
    for factor, group in ic_df.sort_values("date").groupby("factor"):
        g = group[["date", "factor", "rank_ic"]].copy()
        g["rolling_rank_ic_mean"] = g["rank_ic"].rolling(window=window, min_periods=max(3, window // 2)).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
