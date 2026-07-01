from __future__ import annotations

import numpy as np
import pandas as pd

from .performance import TRADING_DAYS


def classify_market_regime(
    benchmark: pd.DataFrame,
    lookback_days: int = 120,
    bull_threshold: float = 0.10,
    bear_threshold: float = -0.10,
) -> pd.DataFrame:
    """Classify bull/bear/sideways regimes from benchmark trend and momentum."""
    df = benchmark.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ret_lookback"] = df["close"].pct_change(lookback_days)
    df["ma"] = df["close"].rolling(lookback_days, min_periods=max(20, lookback_days // 3)).mean()
    df["ma_gap"] = df["close"] / df["ma"] - 1.0
    conditions = [
        (df["ret_lookback"] >= bull_threshold) & (df["ma_gap"] >= 0),
        (df["ret_lookback"] <= bear_threshold) & (df["ma_gap"] <= 0),
    ]
    choices = ["bull", "bear"]
    df["regime"] = np.select(conditions, choices, default="sideways")
    return df[["date", "regime", "ret_lookback", "ma_gap"]]


def summarize_by_regime(backtest: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    df = backtest.merge(regime[["date", "regime"]], on="date", how="left")
    df["regime"] = df["regime"].fillna("unknown")
    rows = []
    for regime_name, group in df.groupby("regime"):
        if group.empty:
            continue
        ret = group["strategy_return"].fillna(0.0)
        bench_ret = group["benchmark_return"].fillna(0.0)
        nav = (1 + ret).cumprod()
        ann_return = (1 + ret.mean()) ** TRADING_DAYS - 1
        ann_vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
        sharpe = ann_return / ann_vol if ann_vol and not np.isnan(ann_vol) else np.nan
        max_dd = float((nav / nav.cummax() - 1).min())
        rows.append(
            {
                "regime": regime_name,
                "days": len(group),
                "strategy_annual_return": ann_return,
                "benchmark_annual_return": (1 + bench_ret.mean()) ** TRADING_DAYS - 1,
                "annual_excess_return": ann_return - ((1 + bench_ret.mean()) ** TRADING_DAYS - 1),
                "annual_volatility": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "win_rate": (ret > 0).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("regime")
