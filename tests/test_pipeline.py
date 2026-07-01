from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mf_strategy.config import DEFAULT_CONFIG, deep_update
from mf_strategy.data_loader import load_market_data
from mf_strategy.factor_engine import build_factor_panel, get_rebalance_dates
from mf_strategy.factor_preprocess import add_category_scores, preprocess_factor_panel
from mf_strategy.factor_test import add_forward_returns, calculate_ic
from mf_strategy.portfolio import build_target_weights
from mf_strategy.backtest import run_backtest
from mf_strategy.performance import summarize_performance


def test_demo_pipeline_smoke():
    cfg = deep_update(
        DEFAULT_CONFIG,
        {
            "data": {"source": "demo", "demo": {"start_date": "2020-01-01", "end_date": "2021-12-31", "n_symbols": 50, "n_industries": 5}},
            "backtest": {"start_date": "2020-09-01", "end_date": "2021-12-31", "top_n": 10, "min_avg_amount": 0},
            "factor": {"neutralize": True, "min_ic_obs": 3},
        },
    )
    data = load_market_data(cfg)
    panel, factor_cols, categories = build_factor_panel(data.prices, data.fundamentals, data.membership, cfg)
    assert not panel.empty
    rebal_dates = get_rebalance_dates(data.prices, cfg["backtest"]["start_date"], cfg["backtest"]["end_date"])
    panel = preprocess_factor_panel(panel, factor_cols, cfg)
    panel = add_forward_returns(panel, data.prices, rebal_dates)
    panel = add_category_scores(panel, categories, cfg["factor"]["category_weights"])
    ic = calculate_ic(panel, factor_cols)
    assert not ic.empty
    weights = build_target_weights(panel, "score_equal", top_n=10, industry_neutral=True)
    assert not weights.empty
    bt = run_backtest(data.prices, data.benchmark, weights, transaction_cost_bps=10, execution_lag=1)
    perf = summarize_performance(bt)
    assert perf["final_nav"].iloc[0] > 0
