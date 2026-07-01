from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemoData:
    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    benchmark: pd.DataFrame
    membership: pd.DataFrame | None = None


def _make_symbols(n_symbols: int) -> list[str]:
    symbols = []
    for i in range(n_symbols):
        exchange = "SH" if i % 2 == 0 else "SZ"
        prefix = "600" if exchange == "SH" else "000"
        symbols.append(f"{prefix}{i + 1:03d}.{exchange}")
    return symbols


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    nav = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(nav)
    drawdown = nav / running_max - 1.0
    return float(drawdown.min())


def generate_demo_data(
    start_date: str = "2017-01-01",
    end_date: str = "2024-12-31",
    n_symbols: int = 180,
    n_industries: int = 9,
    seed: int = 42,
) -> DemoData:
    """Generate realistic-looking data so the full project is runnable without external data.

    The demo data intentionally embeds weak value, quality, momentum and low-volatility premia.
    It is for pipeline validation only and must not be interpreted as market evidence.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    symbols = _make_symbols(n_symbols)
    industries = [f"Industry_{i + 1}" for i in range(n_industries)]

    # Market regimes create bull/bear/sideways periods in the synthetic benchmark.
    regime_cycle = np.sin(np.linspace(0, 5 * np.pi, n_days))
    drift = np.where(regime_cycle > 0.45, 0.00100, np.where(regime_cycle < -0.45, 0.00005, 0.00035))
    market_ret = drift + rng.normal(0.0, 0.010, size=n_days)
    benchmark_close = 1000 * np.cumprod(1 + market_ret)
    benchmark = pd.DataFrame({"date": dates, "symbol": "CSI300_DEMO", "close": benchmark_close})

    symbol_industry = {sym: industries[i % n_industries] for i, sym in enumerate(symbols)}
    rows: list[pd.DataFrame] = []
    fundamental_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []

    monthly_dates = pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).max().tolist()

    for idx, symbol in enumerate(symbols):
        industry = symbol_industry[symbol]
        listed_offset = int(rng.integers(0, min(260, max(1, n_days // 4))))
        listing_date = dates[listed_offset]

        value_score = rng.normal()
        quality_score = 0.35 * value_score + rng.normal()
        liquidity_score = rng.normal()
        size_score = 0.6 * liquidity_score + rng.normal(0, 0.8)
        beta = float(np.clip(rng.normal(1.0, 0.25), 0.45, 1.75))
        idio_vol = float(np.clip(0.012 + 0.006 * rng.random() - 0.0015 * quality_score, 0.006, 0.035))

        # Daily latent stock returns. Premia are weak but enough for demo IC to be non-random.
        style_alpha = (
            0.000120 * value_score
            + 0.000150 * quality_score
            - 0.000040 * idio_vol / 0.015
            + 0.000050 * liquidity_score
        )
        eps = rng.normal(0.0, idio_vol, size=n_days)
        stock_ret = beta * market_ret + style_alpha + eps
        stock_ret[:listed_offset] = np.nan
        close = 12.0 * np.cumprod(1.0 + np.nan_to_num(stock_ret, nan=0.0)) * rng.uniform(0.6, 2.2)
        close[:listed_offset] = np.nan
        open_px = close * (1 + rng.normal(0, 0.003, size=n_days))
        high = np.maximum(open_px, close) * (1 + np.abs(rng.normal(0, 0.006, size=n_days)))
        low = np.minimum(open_px, close) * (1 - np.abs(rng.normal(0, 0.006, size=n_days)))
        base_amount = np.exp(16.5 + 0.45 * liquidity_score + 0.35 * size_score)
        amount = base_amount * (1 + 35 * np.abs(stock_ret)) * rng.lognormal(0.0, 0.35, size=n_days)
        volume = amount / np.maximum(close, 0.1)
        turnover = np.clip(0.008 + 0.01 * rng.random(n_days) + 0.01 * liquidity_score + 0.6 * np.abs(stock_ret), 0.001, 0.25)

        is_tradable = np.ones(n_days, dtype=int)
        suspension_days = rng.choice(np.arange(listed_offset, n_days), size=max(1, n_days // 150), replace=False)
        is_tradable[suspension_days] = 0
        is_st = np.zeros(n_days, dtype=int)
        if rng.random() < 0.06:
            st_start = int(rng.integers(listed_offset, n_days))
            is_st[st_start:] = 1

        df = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": open_px,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
                "turnover_rate": turnover,
                "is_tradable": is_tradable,
                "is_st": is_st,
                "listing_date": listing_date,
            }
        ).dropna(subset=["close"])
        rows.append(df)

        for m_date in monthly_dates:
            if m_date < listing_date:
                continue
            age_years = max((m_date - listing_date).days / 365.25, 0.1)
            noise = rng.normal(0, 0.05)
            pe = np.exp(3.0 - 0.30 * value_score + 0.10 * quality_score + noise)
            pb = np.exp(1.1 - 0.28 * value_score + 0.15 * quality_score + rng.normal(0, 0.08))
            ps = np.exp(1.4 - 0.22 * value_score + rng.normal(0, 0.08))
            roe = np.clip(0.09 + 0.035 * quality_score + 0.01 * rng.normal(), -0.15, 0.35)
            roa = np.clip(0.035 + 0.018 * quality_score + 0.005 * rng.normal(), -0.08, 0.20)
            gross_margin = np.clip(0.28 + 0.05 * quality_score + 0.02 * rng.normal(), 0.03, 0.80)
            net_margin = np.clip(0.08 + 0.025 * quality_score + 0.01 * rng.normal(), -0.20, 0.35)
            debt_to_asset = np.clip(0.50 - 0.07 * quality_score + 0.04 * rng.normal(), 0.05, 0.92)
            dividend_yield = np.clip(0.018 + 0.006 * value_score + 0.003 * rng.normal(), 0.0, 0.08)
            market_cap = np.exp(22.0 + 0.55 * size_score + 0.05 * age_years + rng.normal(0, 0.10))
            fundamental_rows.append(
                {
                    "date": m_date,
                    "symbol": symbol,
                    "pe_ttm": pe,
                    "pb": pb,
                    "ps_ttm": ps,
                    "dividend_yield": dividend_yield,
                    "roe": roe,
                    "roa": roa,
                    "gross_margin": gross_margin,
                    "net_margin": net_margin,
                    "debt_to_asset": debt_to_asset,
                    "market_cap": market_cap,
                    "industry": industry,
                }
            )
            membership_rows.append({"date": m_date, "symbol": symbol, "in_universe": 1})

    prices = pd.concat(rows, ignore_index=True)
    fundamentals = pd.DataFrame(fundamental_rows)
    membership = pd.DataFrame(membership_rows)

    for col in ["date", "listing_date"]:
        if col in prices.columns:
            prices[col] = pd.to_datetime(prices[col])
    fundamentals["date"] = pd.to_datetime(fundamentals["date"])
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    membership["date"] = pd.to_datetime(membership["date"])

    return DemoData(prices=prices, fundamentals=fundamentals, benchmark=benchmark, membership=membership)
