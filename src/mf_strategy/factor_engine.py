from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


FACTOR_CATEGORIES: dict[str, list[str]] = {
    "value": ["value_ep", "value_bp", "value_sp", "value_dividend"],
    "quality": [
        "quality_roe",
        "quality_roa",
        "quality_gross_margin",
        "quality_net_margin",
        "quality_low_debt",
    ],
    "momentum": ["momentum_20d", "momentum_60d", "momentum_120d"],
    "low_volatility": ["low_vol_20d", "low_vol_60d", "low_mdd_60d"],
    "liquidity": ["liquidity_log_amount", "liquidity_low_amihud"],
}


def get_factor_columns() -> list[str]:
    return [factor for factors in FACTOR_CATEGORIES.values() for factor in factors]


def get_rebalance_dates(
    prices: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    freq: str = "M",
) -> list[pd.Timestamp]:
    """Return the last trading date of each rebalance period."""
    dates = pd.Series(pd.to_datetime(prices["date"].drop_duplicates()).sort_values())
    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if dates.empty:
        return []
    periods = dates.dt.to_period(freq)
    return list(dates.groupby(periods).max())


def _rolling_max_drawdown(prices: np.ndarray) -> float:
    if len(prices) == 0 or np.isnan(prices).all():
        return np.nan
    arr = np.asarray(prices, dtype=float)
    running_max = np.maximum.accumulate(arr)
    drawdown = arr / running_max - 1.0
    return float(abs(np.nanmin(drawdown)))


def calculate_price_factors(prices: pd.DataFrame, rebalance_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Calculate technical, momentum, volatility and liquidity factors."""
    if not rebalance_dates:
        return pd.DataFrame()

    df = prices.sort_values(["symbol", "date"]).copy()
    df["ret_1d"] = df.groupby("symbol")["close"].pct_change()

    for window in [20, 60, 120]:
        df[f"ret_{window}d"] = df.groupby("symbol")["close"].pct_change(window)
        df[f"vol_{window}d"] = (
            df.groupby("symbol")["ret_1d"]
            .rolling(window=window, min_periods=max(10, window // 2))
            .std()
            .reset_index(level=0, drop=True)
            * np.sqrt(252)
        )

    df["mdd_60d"] = (
        df.groupby("symbol")["close"]
        .rolling(window=60, min_periods=30)
        .apply(_rolling_max_drawdown, raw=True)
        .reset_index(level=0, drop=True)
    )
    df["turnover_20d"] = (
        df.groupby("symbol")["turnover_rate"]
        .rolling(window=20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["amount_20d"] = (
        df.groupby("symbol")["amount"]
        .rolling(window=20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["amihud_daily"] = df["ret_1d"].abs() / (df["amount"].replace(0, np.nan) / 100_000_000)
    df["amihud_20d"] = (
        df.groupby("symbol")["amihud_daily"]
        .rolling(window=20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )

    keep_cols = [
        "date",
        "symbol",
        "close",
        "listing_date",
        "is_tradable",
        "is_st",
        "ret_20d",
        "ret_60d",
        "ret_120d",
        "vol_20d",
        "vol_60d",
        "mdd_60d",
        "turnover_20d",
        "amount_20d",
        "amihud_20d",
    ]
    rebal_set = set(pd.to_datetime(rebalance_dates))
    out = df.loc[df["date"].isin(rebal_set), keep_cols].copy()
    return out.reset_index(drop=True)


def align_fundamentals_to_rebalance(
    fundamentals: pd.DataFrame,
    symbols: list[str],
    rebalance_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Use latest available fundamental snapshot on or before each rebalance date."""
    aligned_parts = []
    date_frame = pd.DataFrame({"date": pd.to_datetime(rebalance_dates)})
    fundamentals = fundamentals.sort_values(["symbol", "date"])

    for symbol in symbols:
        sub = fundamentals[fundamentals["symbol"] == symbol].sort_values("date")
        if sub.empty:
            continue
        target = date_frame.copy()
        target["symbol"] = symbol
        merged = pd.merge_asof(
            target.sort_values("date"),
            sub.sort_values("date"),
            on="date",
            by="symbol",
            direction="backward",
        )
        aligned_parts.append(merged)

    if not aligned_parts:
        return pd.DataFrame()
    return pd.concat(aligned_parts, ignore_index=True)


def _safe_inverse(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace(0, np.nan)
    s = s.where(s > 0)
    return 1.0 / s


def build_factor_panel(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    membership: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Build a monthly cross-sectional factor panel."""
    bt_cfg = config["backtest"]
    rebalance_dates = get_rebalance_dates(
        prices,
        start_date=bt_cfg.get("start_date"),
        end_date=bt_cfg.get("end_date"),
        freq=bt_cfg.get("rebalance_freq", "M"),
    )
    if len(rebalance_dates) < 2:
        raise ValueError("Not enough rebalance dates. Check date range and price data.")

    price_factors = calculate_price_factors(prices, rebalance_dates)
    symbols = sorted(price_factors["symbol"].dropna().unique().tolist())
    fundamental_panel = align_fundamentals_to_rebalance(fundamentals, symbols, rebalance_dates)

    panel = price_factors.merge(fundamental_panel, on=["date", "symbol"], how="left")

    if membership is not None and not membership.empty:
        m = membership.copy()
        m["date"] = pd.to_datetime(m["date"])
        m = m.sort_values(["symbol", "date"])
        aligned_membership_parts = []
        date_frame = pd.DataFrame({"date": pd.to_datetime(rebalance_dates)})
        for symbol in symbols:
            sub = m[m["symbol"] == symbol].sort_values("date")
            if sub.empty:
                continue
            target = date_frame.copy()
            target["symbol"] = symbol
            merged = pd.merge_asof(target, sub, on="date", by="symbol", direction="backward")
            aligned_membership_parts.append(merged)
        if aligned_membership_parts:
            mem_panel = pd.concat(aligned_membership_parts, ignore_index=True)
            panel = panel.merge(mem_panel[["date", "symbol", "in_universe"]], on=["date", "symbol"], how="left")
            panel = panel[panel["in_universe"].fillna(0).astype(int) == 1]

    # Basic tradability filters.
    min_listed_days = int(bt_cfg.get("min_listed_days", 180))
    min_avg_amount = float(bt_cfg.get("min_avg_amount", 0))
    panel["listed_days"] = (panel["date"] - pd.to_datetime(panel["listing_date"])).dt.days
    panel = panel[
        (panel["is_tradable"].fillna(1).astype(int) == 1)
        & (panel["is_st"].fillna(0).astype(int) == 0)
        & (panel["listed_days"] >= min_listed_days)
        & (panel["amount_20d"].fillna(0) >= min_avg_amount)
    ].copy()

    # Directional factors: every factor is larger-is-better.
    panel["value_ep"] = _safe_inverse(panel["pe_ttm"])
    panel["value_bp"] = _safe_inverse(panel["pb"])
    panel["value_sp"] = _safe_inverse(panel["ps_ttm"])
    panel["value_dividend"] = panel["dividend_yield"]

    panel["quality_roe"] = panel["roe"]
    panel["quality_roa"] = panel["roa"]
    panel["quality_gross_margin"] = panel["gross_margin"]
    panel["quality_net_margin"] = panel["net_margin"]
    panel["quality_low_debt"] = -panel["debt_to_asset"]

    panel["momentum_20d"] = panel["ret_20d"]
    panel["momentum_60d"] = panel["ret_60d"]
    panel["momentum_120d"] = panel["ret_120d"]

    panel["low_vol_20d"] = -panel["vol_20d"]
    panel["low_vol_60d"] = -panel["vol_60d"]
    panel["low_mdd_60d"] = -panel["mdd_60d"]

    panel["liquidity_log_amount"] = np.log1p(panel["amount_20d"].clip(lower=0))
    panel["liquidity_low_amihud"] = -panel["amihud_20d"]
    panel["size_log_mcap"] = np.log(panel["market_cap"].replace(0, np.nan))

    factor_cols = get_factor_columns()
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel, factor_cols, FACTOR_CATEGORIES.copy()
