from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class TusharePaths:
    prices: Path
    fundamentals: Path
    benchmark: Path
    membership: Path
    stock_basic: Path
    index_weight_raw: Path


def yyyymmdd_to_timestamp(value: str | int | float | None) -> pd.Timestamp | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT
    text = str(int(value)) if isinstance(value, float) else str(value)
    if not text or text.lower() == "nan":
        return pd.NaT
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def timestamp_to_yyyymmdd(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _require_tushare():
    try:
        import tushare as ts  # type: ignore
    except ImportError as exc:  # pragma: no cover - only triggered without optional dependency
        raise ImportError(
            "Tushare is not installed. Run `pip install tushare` or `pip install -r requirements.txt`."
        ) from exc
    return ts


def get_tushare_pro(token: str | None = None, token_env: str = "TUSHARE_TOKEN"):
    ts = _require_tushare()
    resolved = token or os.getenv(token_env)
    if not resolved:
        raise ValueError(
            f"Tushare token is missing. Set environment variable {token_env} or pass --token."
        )
    ts.set_token(resolved)
    return ts.pro_api(resolved)


def _call_with_retry(func, retries: int = 3, pause: float = 0.3, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            df = func(**kwargs)
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as exc:  # pragma: no cover - depends on remote API behavior
            last_error = exc
            if attempt < retries - 1:
                time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"Tushare request failed after {retries} attempts: {last_error}")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _read_cached(path: Path, use_cache: bool) -> pd.DataFrame | None:
    if use_cache and path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return None


def resolve_tushare_paths(cache_dir: str | Path) -> TusharePaths:
    base = Path(cache_dir)
    return TusharePaths(
        prices=base / "prices.csv",
        fundamentals=base / "fundamentals.csv",
        benchmark=base / "benchmark.csv",
        membership=base / "membership.csv",
        stock_basic=base / "stock_basic.csv",
        index_weight_raw=base / "index_weight_raw.csv",
    )


def fetch_stock_basic(pro, cache_dir: str | Path, use_cache: bool, retries: int, pause: float) -> pd.DataFrame:
    paths = resolve_tushare_paths(cache_dir)
    cached = _read_cached(paths.stock_basic, use_cache)
    if cached is not None:
        return cached

    fields = "ts_code,symbol,name,area,industry,market,list_date,exchange,list_status"
    df = _call_with_retry(
        pro.stock_basic,
        retries=retries,
        pause=pause,
        exchange="",
        list_status="L",
        fields=fields,
    )
    _write_csv(df, paths.stock_basic)
    return df


def fetch_index_weight(
    pro,
    index_code: str,
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
    use_cache: bool,
    retries: int,
    pause: float,
) -> pd.DataFrame:
    paths = resolve_tushare_paths(cache_dir)
    cached = _read_cached(paths.index_weight_raw, use_cache)
    if cached is not None:
        return cached

    fields = "index_code,con_code,trade_date,weight"
    df = _call_with_retry(
        pro.index_weight,
        retries=retries,
        pause=pause,
        index_code=index_code,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
    )
    _write_csv(df, paths.index_weight_raw)
    return df


def build_universe(
    pro,
    cfg: Mapping[str, Any],
    stock_basic: pd.DataFrame,
    cache_dir: str | Path,
    use_cache: bool,
    retries: int,
    pause: float,
) -> tuple[list[str], pd.DataFrame]:
    universe_cfg = cfg.get("universe", {})
    mode = str(universe_cfg.get("mode", "index_weight")).lower()
    start_date = cfg["start_date"]
    end_date = cfg["end_date"]

    membership = pd.DataFrame()
    if mode == "custom":
        symbols = list(universe_cfg.get("symbols", []))
    elif mode == "stock_basic":
        max_symbols = universe_cfg.get("max_symbols")
        symbols = stock_basic["ts_code"].dropna().sort_values().tolist()
        if max_symbols:
            symbols = symbols[: int(max_symbols)]
    else:
        index_code = universe_cfg.get("index_code", "000300.SH")
        try:
            raw = fetch_index_weight(pro, index_code, start_date, end_date, cache_dir, use_cache, retries, pause)
            if raw.empty:
                raise ValueError("index_weight returned empty data")
            symbols = sorted(raw["con_code"].dropna().unique().tolist())
            membership = raw.rename(columns={"trade_date": "date", "con_code": "symbol"})[
                ["date", "symbol"]
            ].copy()
            membership["date"] = membership["date"].apply(yyyymmdd_to_timestamp)
            membership["in_universe"] = 1
            membership = membership.dropna(subset=["date", "symbol"]).drop_duplicates(["date", "symbol"])
        except Exception as exc:  # pragma: no cover - depends on Tushare permissions
            fallback = int(universe_cfg.get("fallback_max_symbols", 80))
            print(
                f"[WARN] Failed to fetch index_weight for {index_code}: {exc}. "
                f"Fallback to first {fallback} listed stocks from stock_basic."
            )
            symbols = stock_basic["ts_code"].dropna().sort_values().head(fallback).tolist()

    symbols = [s for s in symbols if isinstance(s, str) and s.endswith((".SH", ".SZ", ".BJ"))]
    max_symbols = universe_cfg.get("max_symbols")
    if max_symbols:
        symbols = symbols[: int(max_symbols)]

    if membership.empty:
        membership = pd.DataFrame(
            {
                "date": [pd.Timestamp(start_date)] * len(symbols),
                "symbol": symbols,
                "in_universe": [1] * len(symbols),
            }
        )
    return symbols, membership


def fetch_price_one(
    ts_module,
    pro,
    symbol: str,
    start_date: str,
    end_date: str,
    adj: str,
    use_pro_bar: bool,
    retries: int,
    pause: float,
) -> pd.DataFrame:
    fields = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
    if use_pro_bar:
        return _call_with_retry(
            ts_module.pro_bar,
            retries=retries,
            pause=pause,
            ts_code=symbol,
            start_date=start_date,
            end_date=end_date,
            adj=adj,
            freq="D",
            asset="E",
            fields=fields,
        )
    return _call_with_retry(
        pro.daily,
        retries=retries,
        pause=pause,
        ts_code=symbol,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
    )


def fetch_daily_basic_one(pro, symbol: str, start_date: str, end_date: str, retries: int, pause: float) -> pd.DataFrame:
    fields = (
        "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
        "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
        "free_share,total_mv,circ_mv"
    )
    return _call_with_retry(
        pro.daily_basic,
        retries=retries,
        pause=pause,
        ts_code=symbol,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
    )


def fetch_fina_indicator_one(pro, symbol: str, start_date: str, end_date: str, retries: int, pause: float) -> pd.DataFrame:
    fields = "ts_code,ann_date,end_date,roe,roa,grossprofit_margin,netprofit_margin,debt_to_assets"
    return _call_with_retry(
        pro.fina_indicator,
        retries=retries,
        pause=pause,
        ts_code=symbol,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
    )


def normalize_prices(raw_prices: pd.DataFrame, raw_daily_basic: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    if raw_prices.empty:
        return pd.DataFrame()
    p = raw_prices.copy()
    p = p.rename(columns={"ts_code": "symbol", "trade_date": "date", "vol": "volume"})
    p["date"] = p["date"].apply(yyyymmdd_to_timestamp)
    # Tushare daily/pro_bar amount is usually in thousand RMB. Convert to RMB to match project filters.
    p["amount"] = pd.to_numeric(p.get("amount"), errors="coerce") * 1000.0
    for col in ["open", "high", "low", "close", "volume"]:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")

    db = raw_daily_basic.copy()
    if not db.empty:
        db = db.rename(columns={"ts_code": "symbol", "trade_date": "date"})
        db["date"] = db["date"].apply(yyyymmdd_to_timestamp)
        db = db[[c for c in ["date", "symbol", "turnover_rate"] if c in db.columns]]
        p = p.merge(db, on=["date", "symbol"], how="left")

    sb = stock_basic.copy()
    sb = sb.rename(columns={"ts_code": "symbol", "list_date": "listing_date"})
    if "listing_date" in sb.columns:
        sb["listing_date"] = sb["listing_date"].apply(yyyymmdd_to_timestamp)
    sb["is_st"] = sb.get("name", "").astype(str).str.contains("ST|退", regex=True).astype(int)
    sb = sb[[c for c in ["symbol", "name", "industry", "listing_date", "is_st"] if c in sb.columns]]
    p = p.merge(sb, on="symbol", how="left")
    p["is_tradable"] = 1
    if "turnover_rate" not in p.columns:
        p["turnover_rate"] = pd.NA
    p = p.dropna(subset=["date", "symbol", "close"]).sort_values(["symbol", "date"])
    return p[["date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover_rate", "listing_date", "is_tradable", "is_st"]]


def normalize_fundamentals(
    raw_daily_basic: pd.DataFrame,
    raw_fina: pd.DataFrame,
    stock_basic: pd.DataFrame,
    quality_lag_days: int = 90,
) -> pd.DataFrame:
    if raw_daily_basic.empty:
        return pd.DataFrame()

    db = raw_daily_basic.copy().rename(columns={"ts_code": "symbol", "trade_date": "date"})
    db["date"] = db["date"].apply(yyyymmdd_to_timestamp)
    for col in ["pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv"]:
        if col in db.columns:
            db[col] = pd.to_numeric(db[col], errors="coerce")
    db["dividend_yield"] = db.get("dv_ttm")
    # Tushare total_mv is in 10k RMB. Convert to RMB.
    db["market_cap"] = db.get("total_mv") * 10000.0
    db = db[["date", "symbol", "pe_ttm", "pb", "ps_ttm", "dividend_yield", "market_cap"]]

    fina = raw_fina.copy()
    if not fina.empty:
        fina = fina.rename(
            columns={
                "ts_code": "symbol",
                "ann_date": "date",
                "grossprofit_margin": "gross_margin",
                "netprofit_margin": "net_margin",
                "debt_to_assets": "debt_to_asset",
            }
        )
        fina["date"] = fina["date"].apply(yyyymmdd_to_timestamp)
        if fina["date"].isna().any() and "end_date" in fina.columns:
            fallback_date = fina["end_date"].apply(yyyymmdd_to_timestamp) + pd.to_timedelta(quality_lag_days, unit="D")
            fina["date"] = fina["date"].fillna(fallback_date)
        for col in ["roe", "roa", "gross_margin", "net_margin", "debt_to_asset"]:
            if col in fina.columns:
                fina[col] = pd.to_numeric(fina[col], errors="coerce")
        fina = fina[["date", "symbol", "roe", "roa", "gross_margin", "net_margin", "debt_to_asset"]]
        fina = fina.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"])

        parts = []
        for symbol, sub_db in db.sort_values(["symbol", "date"]).groupby("symbol"):
            sub_fi = fina[fina["symbol"] == symbol].sort_values("date")
            if sub_fi.empty:
                tmp = sub_db.copy()
                for col in ["roe", "roa", "gross_margin", "net_margin", "debt_to_asset"]:
                    tmp[col] = pd.NA
            else:
                tmp = pd.merge_asof(
                    sub_db.sort_values("date"),
                    sub_fi.sort_values("date"),
                    on="date",
                    by="symbol",
                    direction="backward",
                )
            parts.append(tmp)
        db = pd.concat(parts, ignore_index=True)
    else:
        for col in ["roe", "roa", "gross_margin", "net_margin", "debt_to_asset"]:
            db[col] = pd.NA

    sb = stock_basic.copy().rename(columns={"ts_code": "symbol"})
    sb = sb[[c for c in ["symbol", "industry"] if c in sb.columns]]
    db = db.merge(sb, on="symbol", how="left")
    db["industry"] = db["industry"].fillna("UNKNOWN")
    return db.sort_values(["symbol", "date"])


def fetch_benchmark(pro, index_code: str, start_date: str, end_date: str, retries: int, pause: float) -> pd.DataFrame:
    fields = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
    df = _call_with_retry(
        pro.index_daily,
        retries=retries,
        pause=pause,
        ts_code=index_code,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
    )
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"ts_code": "symbol", "trade_date": "date", "vol": "volume"})
    df["date"] = df["date"].apply(yyyymmdd_to_timestamp)
    df = df.sort_values("date")
    return df[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]


def fetch_tushare_to_csv(config: Mapping[str, Any], token: str | None = None, force: bool = False) -> TusharePaths:
    """Download Tushare data and save normalized CSV files for the backtest pipeline."""
    data_cfg = config.get("data", {})
    ts_cfg = data_cfg.get("tushare", {})
    csv_cfg = data_cfg.get("csv", {})
    cache_dir = ts_cfg.get("cache_dir") or Path(csv_cfg.get("prices_path", "data/raw/tushare/prices.csv")).parent
    cache_dir = Path(cache_dir)
    paths = resolve_tushare_paths(cache_dir)

    use_cache = bool(ts_cfg.get("use_cache", True)) and not force
    retries = int(ts_cfg.get("retries", 3))
    pause = float(ts_cfg.get("request_pause", 0.35))
    start_date = timestamp_to_yyyymmdd(ts_cfg.get("start_date", config.get("backtest", {}).get("start_date", "2018-01-01")))
    end_date = timestamp_to_yyyymmdd(ts_cfg.get("end_date", config.get("backtest", {}).get("end_date", pd.Timestamp.today())))
    adj = ts_cfg.get("adj", "qfq")
    use_pro_bar = bool(ts_cfg.get("use_pro_bar", True))
    token_env = ts_cfg.get("token_env", "TUSHARE_TOKEN")
    benchmark_index = ts_cfg.get("benchmark_index", config.get("backtest", {}).get("benchmark_code", "000300.SH"))
    quality_lag_days = int(ts_cfg.get("quality_lag_days", 90))

    if use_cache and all(path.exists() and path.stat().st_size > 0 for path in [paths.prices, paths.fundamentals, paths.benchmark, paths.membership]):
        print(f"[CACHE] Normalized Tushare CSV files already exist under {cache_dir}")
        return paths

    ts_module = _require_tushare()
    pro = get_tushare_pro(token=token, token_env=token_env)

    print("[Tushare] Fetching stock_basic...")
    stock_basic = fetch_stock_basic(pro, cache_dir, use_cache=use_cache, retries=retries, pause=pause)

    print("[Tushare] Resolving universe...")
    symbols, membership = build_universe(
        pro,
        ts_cfg | {"start_date": start_date, "end_date": end_date},
        stock_basic,
        cache_dir,
        use_cache=use_cache,
        retries=retries,
        pause=pause,
    )
    if not symbols:
        raise ValueError("No symbols resolved for Tushare universe. Check configs/config_tushare.yaml.")
    print(f"[Tushare] Universe size: {len(symbols)}")

    raw_price_parts: list[pd.DataFrame] = []
    raw_basic_parts: list[pd.DataFrame] = []
    raw_fina_parts: list[pd.DataFrame] = []

    raw_dir = cache_dir / "raw_by_symbol"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for symbol in tqdm(symbols, desc="Downloading per-symbol data"):
        price_path = raw_dir / f"price_{symbol}.csv"
        basic_path = raw_dir / f"daily_basic_{symbol}.csv"
        fina_path = raw_dir / f"fina_indicator_{symbol}.csv"

        price = _read_cached(price_path, use_cache)
        if price is None:
            price = fetch_price_one(ts_module, pro, symbol, start_date, end_date, adj, use_pro_bar, retries, pause)
            _write_csv(price, price_path)
            time.sleep(pause)
        raw_price_parts.append(price)

        basic = _read_cached(basic_path, use_cache)
        if basic is None:
            basic = fetch_daily_basic_one(pro, symbol, start_date, end_date, retries, pause)
            _write_csv(basic, basic_path)
            time.sleep(pause)
        raw_basic_parts.append(basic)

        fina = _read_cached(fina_path, use_cache)
        if fina is None:
            fina = fetch_fina_indicator_one(pro, symbol, start_date, end_date, retries, pause)
            _write_csv(fina, fina_path)
            time.sleep(pause)
        raw_fina_parts.append(fina)

    raw_prices = pd.concat([df for df in raw_price_parts if not df.empty], ignore_index=True) if raw_price_parts else pd.DataFrame()
    raw_daily_basic = pd.concat([df for df in raw_basic_parts if not df.empty], ignore_index=True) if raw_basic_parts else pd.DataFrame()
    raw_fina = pd.concat([df for df in raw_fina_parts if not df.empty], ignore_index=True) if raw_fina_parts else pd.DataFrame()

    print("[Tushare] Normalizing price/fundamental/benchmark files...")
    prices = normalize_prices(raw_prices, raw_daily_basic, stock_basic)
    fundamentals = normalize_fundamentals(raw_daily_basic, raw_fina, stock_basic, quality_lag_days=quality_lag_days)
    benchmark = fetch_benchmark(pro, benchmark_index, start_date, end_date, retries, pause)

    if prices.empty:
        raise ValueError("Downloaded price data is empty. Check Tushare permissions and date range.")
    if fundamentals.empty:
        raise ValueError("Downloaded fundamental data is empty. Check daily_basic permissions.")
    if benchmark.empty:
        raise ValueError("Downloaded benchmark data is empty. Check benchmark_index setting.")

    _write_csv(prices, paths.prices)
    _write_csv(fundamentals, paths.fundamentals)
    _write_csv(benchmark, paths.benchmark)
    _write_csv(membership, paths.membership)

    print(f"[Tushare] Saved normalized CSV files under {cache_dir}")
    return paths
