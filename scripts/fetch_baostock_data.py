from __future__ import annotations

"""Fetch free A-share data from BaoStock and convert it to project-standard CSV files.

This script is designed as a no-token fallback data source for the multi-factor
research framework. It writes the same three CSVs that the main pipeline already
knows how to read:

- prices.csv
- fundamentals.csv
- benchmark.csv
- membership.csv

Usage:
    python scripts/fetch_baostock_data.py --config configs/config_baostock.yaml

Notes:
    BaoStock does not provide the same complete point-in-time fundamental set as
    commercial data vendors. This script therefore builds a practical free-data
    research dataset using daily adjusted prices, valuation fields, turnover,
    and optional quarterly financial fields when available.
"""

import argparse
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

try:
    import baostock as bs
except ImportError as exc:  # pragma: no cover - user environment issue
    raise SystemExit(
        "BaoStock is not installed. Please run: pip install baostock"
    ) from exc


HISTORY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,"
    "peTTM,pbMRQ,psTTM"
)

BENCHMARK_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"

STANDARD_FUNDAMENTAL_COLUMNS = [
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
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def bs_code_to_symbol(code: str) -> str:
    """Convert BaoStock code such as sh.600519 to project symbol 600519.SH."""
    market, ticker = code.split(".")
    return f"{ticker}.{market.upper()}"


def symbol_to_bs_code(symbol: str) -> str:
    """Convert project symbol such as 600519.SH to BaoStock code sh.600519."""
    ticker, market = symbol.split(".")
    return f"{market.lower()}.{ticker}"


def _query_to_dataframe(rs) -> pd.DataFrame:
    rows: list[list[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        print(f"[WARN] BaoStock query error: {rs.error_code} {rs.error_msg}")
    return pd.DataFrame(rows, columns=rs.fields)


def _to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _infer_market_cap(close: pd.Series, volume: pd.Series, turnover_rate: pd.Series, amount: pd.Series) -> pd.Series:
    """Infer a rough float-market-cap proxy from free data.

    BaoStock daily data includes volume and turnover. If turnover is available,
    free-float shares can be approximated as volume / (turnover_rate / 100), so
    market_cap ~= close * free_float_shares. This is not a replacement for a
    vendor-grade market cap field, but is adequate for size neutralization in a
    free-data educational project.
    """
    close = pd.to_numeric(close, errors="coerce")
    volume = pd.to_numeric(volume, errors="coerce")
    turnover_rate = pd.to_numeric(turnover_rate, errors="coerce")
    amount = pd.to_numeric(amount, errors="coerce")

    valid_turn = turnover_rate.where(turnover_rate > 0)
    free_float_shares = volume / (valid_turn / 100.0)
    mcap = close * free_float_shares

    # Fallback: use a scaled amount proxy where turnover is missing.
    fallback = amount.rolling(20, min_periods=1).mean() * 20.0
    return mcap.where(np.isfinite(mcap) & (mcap > 0), fallback)


def fetch_stock_basic(end_date: str, max_symbols: int | None, custom_symbols: list[str] | None = None) -> pd.DataFrame:
    if custom_symbols:
        records = []
        for symbol in custom_symbols:
            records.append(
                {
                    "symbol": symbol,
                    "bs_code": symbol_to_bs_code(symbol),
                    "name": symbol,
                    "listing_date": "2000-01-01",
                    "is_st": 0,
                    "industry": "Unknown",
                }
            )
        return pd.DataFrame(records)

    print("[BaoStock] Fetching stock list...")
    rs = bs.query_all_stock(day=end_date)
    df = _query_to_dataframe(rs)
    if df.empty:
        raise RuntimeError("BaoStock returned an empty stock list. Please check date or network.")

    # Keep common A-share tickers; exclude indices/funds/other instruments.
    df = df[df["code"].str.startswith(("sh.6", "sz.0", "sz.3"))].copy()
    if "code_name" not in df.columns:
        df["code_name"] = df["code"]

    basic_rs = bs.query_stock_basic()
    basic = _query_to_dataframe(basic_rs)
    if not basic.empty:
        df = df.merge(basic, on="code", how="left", suffixes=("", "_basic"))

    name_col = "code_name"
    if "code_name_basic" in df.columns:
        df[name_col] = df["code_name_basic"].fillna(df[name_col])

    out = pd.DataFrame(
        {
            "symbol": df["code"].map(bs_code_to_symbol),
            "bs_code": df["code"],
            "name": df[name_col].fillna(df["code"]),
            "listing_date": df.get("ipoDate", pd.Series("2000-01-01", index=df.index)).replace("", np.nan).fillna("2000-01-01"),
            "is_st": df[name_col].fillna("").str.contains("ST", case=False, regex=False).astype(int),
            "industry": "Unknown",
        }
    )
    out = out.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    if max_symbols is not None:
        out = out.head(int(max_symbols)).copy()
    return out


def fetch_stock_history(bs_code: str, start_date: str, end_date: str, adjustflag: str, retries: int, pause: float) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, retries + 1):
        rs = bs.query_history_k_data_plus(
            bs_code,
            HISTORY_FIELDS,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag,
        )
        df = _query_to_dataframe(rs)
        if rs.error_code == "0":
            return df
        last_error = f"{rs.error_code} {rs.error_msg}"
        time.sleep(pause * attempt)
    print(f"[WARN] Failed to fetch {bs_code}: {last_error}")
    return pd.DataFrame()


def fetch_benchmark(benchmark_code: str, start_date: str, end_date: str, pause: float) -> pd.DataFrame:
    print(f"[BaoStock] Fetching benchmark {benchmark_code}...")
    rs = bs.query_history_k_data_plus(
        benchmark_code,
        BENCHMARK_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    df = _query_to_dataframe(rs)
    if df.empty:
        raise RuntimeError(f"BaoStock returned empty benchmark data for {benchmark_code}.")
    df = _to_float(df, ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "symbol": bs_code_to_symbol(benchmark_code),
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df["volume"],
            "amount": df["amount"],
        }
    )
    time.sleep(pause)
    return out


def fetch_optional_financials(symbols: list[str], cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Fetch quarterly profit/balance data if available.

    This is optional because BaoStock financial interfaces are quarterly and can
    be slower. Missing fields are fine; the script falls back to neutral default
    quality values in fundamentals.csv.
    """
    if not bool(cfg.get("fetch_financials", False)):
        return pd.DataFrame()

    start_year = int(str(cfg["start_date"])[:4])
    end_year = int(str(cfg["end_date"])[:4])
    years = range(start_year, end_year + 1)
    quarters = [1, 2, 3, 4]
    pause = float(cfg.get("request_pause", 0.2))
    max_symbols = int(cfg.get("financial_max_symbols", len(symbols)))
    parts: list[pd.DataFrame] = []

    print(f"[BaoStock] Fetching optional quarterly financials for up to {max_symbols} symbols...")
    for symbol in tqdm(symbols[:max_symbols], desc="Financials"):
        bs_code = symbol_to_bs_code(symbol)
        for year in years:
            for quarter in quarters:
                profit = _query_to_dataframe(bs.query_profit_data(code=bs_code, year=year, quarter=quarter))
                balance = _query_to_dataframe(bs.query_balance_data(code=bs_code, year=year, quarter=quarter))
                if profit.empty and balance.empty:
                    continue
                merged = profit
                if not balance.empty:
                    if merged.empty:
                        merged = balance
                    else:
                        keys = [c for c in ["code", "pubDate", "statDate"] if c in merged.columns and c in balance.columns]
                        merged = merged.merge(balance, on=keys, how="outer", suffixes=("", "_bal"))
                if not merged.empty:
                    merged["symbol"] = symbol
                    parts.append(merged)
                time.sleep(pause)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_standard_csvs(stock_basic: pd.DataFrame, histories: list[pd.DataFrame], benchmark: pd.DataFrame, output_dir: Path) -> None:
    if not histories:
        raise RuntimeError("No stock history was fetched. Please reduce filters or check BaoStock connection.")

    raw = pd.concat(histories, ignore_index=True)
    raw = _to_float(raw, ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "peTTM", "pbMRQ", "psTTM"])
    raw["date"] = pd.to_datetime(raw["date"])
    raw["symbol"] = raw["code"].map(bs_code_to_symbol)

    meta = stock_basic[["symbol", "listing_date", "is_st", "industry"]].copy()
    meta["listing_date"] = pd.to_datetime(meta["listing_date"], errors="coerce").fillna(pd.Timestamp("2000-01-01"))
    raw = raw.merge(meta, on="symbol", how="left")

    prices = pd.DataFrame(
        {
            "date": raw["date"],
            "symbol": raw["symbol"],
            "open": raw["open"],
            "high": raw["high"],
            "low": raw["low"],
            "close": raw["close"],
            "volume": raw["volume"],
            "amount": raw["amount"],
            "turnover_rate": raw["turn"],
            "listing_date": raw["listing_date"],
            "is_tradable": 1,
            "is_st": raw["is_st"].fillna(0).astype(int),
        }
    )

    market_cap = raw.groupby("symbol", group_keys=False).apply(
        lambda g: _infer_market_cap(g["close"], g["volume"], g["turn"], g["amount"])
    )
    market_cap = market_cap.reset_index(level=0, drop=True) if isinstance(market_cap.index, pd.MultiIndex) else market_cap

    fundamentals = pd.DataFrame(
        {
            "date": raw["date"],
            "symbol": raw["symbol"],
            "pe_ttm": raw["peTTM"],
            "pb": raw["pbMRQ"],
            "ps_ttm": raw["psTTM"],
            "dividend_yield": 0.0,
            # BaoStock daily history does not include these point-in-time quality metrics.
            # Neutral defaults keep the full project pipeline runnable; quality factors will
            # contribute little unless the user later adds vendor-grade financial data.
            "roe": 0.0,
            "roa": 0.0,
            "gross_margin": 0.0,
            "net_margin": 0.0,
            "debt_to_asset": 0.0,
            "market_cap": market_cap.to_numpy(),
            "industry": raw["industry"].fillna("Unknown"),
        }
    )

    membership = prices[["date", "symbol"]].drop_duplicates().copy()
    membership["in_universe"] = 1

    output_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_dir / "prices.csv", index=False, encoding="utf-8-sig")
    fundamentals[STANDARD_FUNDAMENTAL_COLUMNS].to_csv(output_dir / "fundamentals.csv", index=False, encoding="utf-8-sig")
    benchmark.to_csv(output_dir / "benchmark.csv", index=False, encoding="utf-8-sig")
    membership.to_csv(output_dir / "membership.csv", index=False, encoding="utf-8-sig")
    stock_basic.to_csv(output_dir / "stock_basic.csv", index=False, encoding="utf-8-sig")

    print(f"[BaoStock] Saved standard CSVs to {output_dir}")
    print(f"[BaoStock] prices rows: {len(prices):,}, fundamentals rows: {len(fundamentals):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch free A-share data from BaoStock.")
    parser.add_argument("--config", default="configs/config_baostock.yaml", help="Path to YAML config.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing standard CSV files.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    bs_cfg = config.get("data", {}).get("baostock", {})
    output_dir = Path(bs_cfg.get("output_dir", "data/raw/baostock"))
    required_files = ["prices.csv", "fundamentals.csv", "benchmark.csv", "membership.csv"]
    if not args.force and all((output_dir / f).exists() for f in required_files):
        print(f"[BaoStock] Standard CSV files already exist in {output_dir}. Use --force to refresh.")
        return

    start_date = str(bs_cfg.get("start_date", "2020-01-01"))
    end_date = str(bs_cfg.get("end_date", "2023-12-31"))
    max_symbols = bs_cfg.get("max_symbols", 80)
    custom_symbols = bs_cfg.get("symbols")
    benchmark_code = str(bs_cfg.get("benchmark_code", "sh.000300"))
    adjustflag = str(bs_cfg.get("adjustflag", "2"))
    pause = float(bs_cfg.get("request_pause", 0.2))
    retries = int(bs_cfg.get("retries", 3))

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")

    try:
        stock_basic = fetch_stock_basic(end_date=end_date, max_symbols=max_symbols, custom_symbols=custom_symbols)
        print(f"[BaoStock] Universe size: {len(stock_basic)}")

        histories: list[pd.DataFrame] = []
        for row in tqdm(stock_basic.itertuples(index=False), total=len(stock_basic), desc="Downloading BaoStock data"):
            df = fetch_stock_history(row.bs_code, start_date, end_date, adjustflag, retries, pause)
            if not df.empty:
                histories.append(df)
            time.sleep(pause)

        benchmark = fetch_benchmark(benchmark_code, start_date, end_date, pause)
        build_standard_csvs(stock_basic, histories, benchmark, output_dir)
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
