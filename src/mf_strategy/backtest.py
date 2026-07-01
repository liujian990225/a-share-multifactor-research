from __future__ import annotations

import numpy as np
import pandas as pd


def _next_trading_date(dates: pd.Index, signal_date: pd.Timestamp, lag: int = 1) -> pd.Timestamp | None:
    pos = dates.searchsorted(pd.Timestamp(signal_date), side="right") + max(lag - 1, 0)
    if pos >= len(dates):
        return None
    return pd.Timestamp(dates[pos])


def run_backtest(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    target_weights: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
    execution_lag: int = 1,
) -> pd.DataFrame:
    """Run an equal-weight portfolio backtest with transaction costs.

    Rebalance signal is produced on the rebalance date and executed after `execution_lag`
    trading days. Costs are deducted on execution date based on one-way turnover.
    """
    if target_weights.empty:
        raise ValueError("target_weights is empty. No portfolio can be backtested.")

    price_pivot = prices.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    returns = price_pivot.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    trading_dates = returns.index

    bench = benchmark.sort_values("date").set_index("date")["close"].reindex(trading_dates).ffill()
    bench_ret = bench.pct_change().fillna(0.0)

    execution_map: dict[pd.Timestamp, pd.Series] = {}
    for signal_date, group in target_weights.groupby("date"):
        exec_date = _next_trading_date(trading_dates, pd.Timestamp(signal_date), execution_lag)
        if exec_date is None:
            continue
        weights = group.set_index("symbol")["weight"].astype(float)
        weights = weights / weights.sum()
        execution_map[exec_date] = weights

    current_weights = pd.Series(dtype=float)
    cost_rate = transaction_cost_bps / 10_000.0
    rows = []
    nav = 1.0
    benchmark_nav = 1.0

    for date in trading_dates:
        turnover = 0.0
        cost = 0.0
        if date in execution_map:
            new_weights = execution_map[date]
            all_symbols = current_weights.index.union(new_weights.index)
            old = current_weights.reindex(all_symbols).fillna(0.0)
            new = new_weights.reindex(all_symbols).fillna(0.0)
            turnover = float((new - old).abs().sum())
            cost = turnover * cost_rate
            current_weights = new_weights

        day_ret = 0.0
        if not current_weights.empty:
            aligned_ret = returns.loc[date].reindex(current_weights.index).fillna(0.0)
            day_ret = float((current_weights * aligned_ret).sum())
        strategy_ret = day_ret - cost
        nav *= 1.0 + strategy_ret
        benchmark_nav *= 1.0 + float(bench_ret.loc[date])
        rows.append(
            {
                "date": date,
                "strategy_return": strategy_ret,
                "benchmark_return": float(bench_ret.loc[date]),
                "turnover": turnover,
                "transaction_cost": cost,
                "nav": nav,
                "benchmark_nav": benchmark_nav,
            }
        )

    result = pd.DataFrame(rows)
    result["excess_return"] = result["strategy_return"] - result["benchmark_return"]
    result["excess_nav"] = (1 + result["excess_return"]).cumprod()
    result["drawdown"] = result["nav"] / result["nav"].cummax() - 1.0
    result["benchmark_drawdown"] = result["benchmark_nav"] / result["benchmark_nav"].cummax() - 1.0
    return result


def run_sensitivity_grid(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    panel: pd.DataFrame,
    score_col: str,
    build_weights_func,
    performance_func,
    top_n_list: list[int],
    cost_bps_list: list[float],
    execution_lag: int = 1,
    industry_neutral: bool = True,
) -> pd.DataFrame:
    """Evaluate strategy performance under different top-N and cost assumptions."""
    rows = []
    for top_n in top_n_list:
        weights = build_weights_func(
            panel,
            score_col=score_col,
            top_n=top_n,
            top_pct=None,
            industry_neutral=industry_neutral,
        )
        for cost_bps in cost_bps_list:
            bt = run_backtest(
                prices,
                benchmark,
                weights,
                transaction_cost_bps=cost_bps,
                execution_lag=execution_lag,
            )
            perf = performance_func(bt)
            row = perf.iloc[0].to_dict()
            row.update({"score": score_col, "top_n": top_n, "cost_bps": cost_bps})
            rows.append(row)
    return pd.DataFrame(rows)
