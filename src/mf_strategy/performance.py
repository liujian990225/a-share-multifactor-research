from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _annualized_return(nav: pd.Series) -> float:
    nav = nav.dropna()
    if len(nav) < 2:
        return np.nan
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    years = len(nav) / TRADING_DAYS
    if years <= 0:
        return np.nan
    return (1.0 + total) ** (1.0 / years) - 1.0


def _max_drawdown(nav: pd.Series) -> float:
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min())


def summarize_performance(backtest: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """Return one-row performance summary."""
    df = backtest.copy()
    ret = df["strategy_return"].fillna(0.0)
    bench_ret = df["benchmark_return"].fillna(0.0)
    excess = ret - bench_ret

    ann_return = _annualized_return(df["nav"])
    ann_vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    downside = ret[ret < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol and not np.isnan(ann_vol) else np.nan
    sortino = (ann_return - risk_free_rate) / downside if downside and not np.isnan(downside) else np.nan
    max_dd = _max_drawdown(df["nav"])
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    bench_ann_return = _annualized_return(df["benchmark_nav"])
    tracking_error = excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    info_ratio = (ann_return - bench_ann_return) / tracking_error if tracking_error and not np.isnan(tracking_error) else np.nan

    beta = np.nan
    alpha = np.nan
    if bench_ret.var(ddof=1) > 0:
        beta = ret.cov(bench_ret) / bench_ret.var(ddof=1)
        alpha_daily = ret.mean() - beta * bench_ret.mean()
        alpha = (1 + alpha_daily) ** TRADING_DAYS - 1

    monthly = df.set_index("date")["strategy_return"].resample("ME").apply(lambda s: (1 + s).prod() - 1)
    monthly_win_rate = (monthly > 0).mean() if not monthly.empty else np.nan

    return pd.DataFrame(
        [
            {
                "annual_return": ann_return,
                "benchmark_annual_return": bench_ann_return,
                "annual_excess_return": ann_return - bench_ann_return,
                "annual_volatility": ann_vol,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": max_dd,
                "calmar": calmar,
                "information_ratio": info_ratio,
                "beta": beta,
                "alpha": alpha,
                "daily_win_rate": (ret > 0).mean(),
                "monthly_win_rate": monthly_win_rate,
                "avg_turnover": df["turnover"].mean(),
                "total_transaction_cost": df["transaction_cost"].sum(),
                "final_nav": df["nav"].iloc[-1],
                "benchmark_final_nav": df["benchmark_nav"].iloc[-1],
            }
        ]
    )


def yearly_returns(backtest: pd.DataFrame) -> pd.DataFrame:
    df = backtest.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    rows = []
    for year, group in df.groupby("year"):
        rows.append(
            {
                "year": year,
                "strategy_return": (1 + group["strategy_return"]).prod() - 1,
                "benchmark_return": (1 + group["benchmark_return"]).prod() - 1,
                "excess_return": (1 + group["excess_return"]).prod() - 1,
                "max_drawdown": _max_drawdown((1 + group["strategy_return"]).cumprod()),
                "turnover": group["turnover"].sum(),
            }
        )
    return pd.DataFrame(rows)


def monthly_return_table(backtest: pd.DataFrame) -> pd.DataFrame:
    df = backtest.copy()
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date")["strategy_return"].resample("ME").apply(lambda s: (1 + s).prod() - 1)
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot(index="year", columns="month", values="return")
    return pivot.sort_index()
