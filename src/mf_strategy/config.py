from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "a_share_multifactor", "random_seed": 42},
    "data": {
        "source": "demo",
        "demo": {
            "start_date": "2017-01-01",
            "end_date": "2024-12-31",
            "n_symbols": 180,
            "n_industries": 9,
        },
        "csv": {
            "prices_path": "data/raw/prices.csv",
            "fundamentals_path": "data/raw/fundamentals.csv",
            "benchmark_path": "data/raw/benchmark.csv",
            "membership_path": None,
        },
    },
    "backtest": {
        "start_date": "2018-01-01",
        "end_date": "2024-12-31",
        "rebalance_freq": "M",
        "execution_lag": 1,
        "top_n": 30,
        "top_pct": None,
        "transaction_cost_bps": 10,
        "benchmark_name": "CSI300",
        "industry_neutral_selection": True,
        "min_listed_days": 180,
        "min_avg_amount": 20_000_000,
    },
    "factor": {
        "neutralize": True,
        "neutralize_size": True,
        "winsorize_method": "mad",
        "mad_n": 3.0,
        "ic_lookback": 12,
        "min_ic_obs": 6,
        "quantiles": 5,
        "category_weights": {
            "value": 0.25,
            "quality": 0.25,
            "momentum": 0.20,
            "low_volatility": 0.15,
            "liquidity": 0.15,
        },
    },
    "analysis": {
        "sensitivity": {"top_n_list": [20, 30, 50], "cost_bps_list": [5, 10, 20]},
        "regime": {"lookback_days": 120, "bull_threshold": 0.10, "bear_threshold": -0.10},
    },
    "ml": {
        "enabled": False,
        "model_type": "random_forest",
        "n_estimators": 200,
        "max_depth": 3,
        "train_window_months": 36,
        "min_train_months": 18,
        "validation_months": 6,
        "rank_target": True,
        "factor_ic_forecast": True,
        "ic_model_type": "random_forest",
        "ic_feature_lookback": 12,
        "ic_min_train_rows": 80,
    },
    "reports": {"output_dir": "reports/demo_run", "save_figures": True, "save_csv": True},
}


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update a dictionary without mutating the input."""
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and merge it with defaults."""
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    return deep_update(config, user_config)


def ensure_output_dirs(config: Mapping[str, Any]) -> dict[str, Path]:
    """Create report folders and return useful paths."""
    output_dir = Path(config["reports"]["output_dir"])
    data_dir = output_dir / "data"
    fig_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {"output": output_dir, "data": data_dir, "figures": fig_dir}
