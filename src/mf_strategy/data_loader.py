from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .demo_data import DemoData, generate_demo_data


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    benchmark: pd.DataFrame
    membership: pd.DataFrame | None = None


PRICE_REQUIRED = {"date", "symbol", "open", "high", "low", "close", "volume", "amount"}
FUNDAMENTAL_REQUIRED = {
    "date",
    "symbol",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dividend_yield",
    "roe",
    "roa",
    "gross_margin",
    "net_margin",
    "debt_to_asset",
    "market_cap",
    "industry",
}
BENCHMARK_REQUIRED = {"date", "close"}


def _read_csv(path: str | Path, date_cols: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    df = pd.read_csv(path)
    for col in date_cols or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    if "listing_date" in prices.columns:
        prices["listing_date"] = pd.to_datetime(prices["listing_date"])
    else:
        prices["listing_date"] = prices.groupby("symbol")["date"].transform("min")
    if "turnover_rate" not in prices.columns:
        prices["turnover_rate"] = prices["volume"] / prices.groupby("symbol")["volume"].transform(lambda s: s.rolling(252, min_periods=20).mean())
    if "is_tradable" not in prices.columns:
        prices["is_tradable"] = 1
    if "is_st" not in prices.columns:
        prices["is_st"] = 0
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    return prices


def _clean_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    fundamentals = fundamentals.copy()
    fundamentals["date"] = pd.to_datetime(fundamentals["date"])
    fundamentals = fundamentals.sort_values(["symbol", "date"]).reset_index(drop=True)
    return fundamentals


def _clean_benchmark(benchmark: pd.DataFrame) -> pd.DataFrame:
    benchmark = benchmark.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    if "symbol" not in benchmark.columns:
        benchmark["symbol"] = "BENCHMARK"
    benchmark = benchmark.sort_values("date").reset_index(drop=True)
    return benchmark


def load_market_data(config: Mapping[str, Any]) -> MarketData:
    """Load market data from demo generator or CSV files."""
    source = config["data"].get("source", "demo").lower()
    seed = int(config.get("project", {}).get("random_seed", 42))

    if source == "demo":
        demo_cfg = config["data"].get("demo", {})
        data: DemoData = generate_demo_data(
            start_date=demo_cfg.get("start_date", "2017-01-01"),
            end_date=demo_cfg.get("end_date", "2024-12-31"),
            n_symbols=int(demo_cfg.get("n_symbols", 180)),
            n_industries=int(demo_cfg.get("n_industries", 9)),
            seed=seed,
        )
        return MarketData(
            prices=_clean_prices(data.prices),
            fundamentals=_clean_fundamentals(data.fundamentals),
            benchmark=_clean_benchmark(data.benchmark),
            membership=data.membership,
        )

    if source == "csv":
        csv_cfg = config["data"].get("csv", {})
        prices = _read_csv(csv_cfg["prices_path"], date_cols=["date", "listing_date"])
        fundamentals = _read_csv(csv_cfg["fundamentals_path"], date_cols=["date"])
        benchmark = _read_csv(csv_cfg["benchmark_path"], date_cols=["date"])
        membership = None
        if csv_cfg.get("membership_path"):
            membership_path = Path(csv_cfg["membership_path"])
            if membership_path.exists():
                membership = _read_csv(membership_path, date_cols=["date"])

        _validate_columns(prices, PRICE_REQUIRED, "prices.csv")
        _validate_columns(fundamentals, FUNDAMENTAL_REQUIRED, "fundamentals.csv")
        _validate_columns(benchmark, BENCHMARK_REQUIRED, "benchmark.csv")

        return MarketData(
            prices=_clean_prices(prices),
            fundamentals=_clean_fundamentals(fundamentals),
            benchmark=_clean_benchmark(benchmark),
            membership=membership,
        )

    raise ValueError(f"Unsupported data source: {source}. Use 'demo' or 'csv'.")
