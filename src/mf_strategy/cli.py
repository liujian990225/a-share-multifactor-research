from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .backtest import run_backtest, run_sensitivity_grid
from .config import ensure_output_dirs, load_config
from .data_loader import load_market_data
from .factor_engine import build_factor_panel, get_rebalance_dates
from .factor_preprocess import add_category_scores, add_rolling_ic_weighted_score, preprocess_factor_panel
from .factor_test import (
    add_forward_returns,
    calculate_ic,
    factor_correlation,
    quantile_analysis,
    rolling_factor_decay,
    summarize_ic,
)
from .market_regime import classify_market_regime, summarize_by_regime
from .ml_alpha import add_factor_ic_forecast_score, add_walk_forward_ml_alpha_score
from .performance import monthly_return_table, summarize_performance, yearly_returns
from .portfolio import build_target_weights, calculate_industry_exposure, calculate_style_exposure
from .report import write_markdown_report
from .visualization import (
    plot_drawdown,
    plot_factor_correlation,
    plot_ic,
    plot_industry_exposure,
    plot_feature_importance,
    plot_factor_timing_weights,
    plot_monthly_heatmap,
    plot_nav,
    plot_quantile_returns,
)


def _save_csv(df: pd.DataFrame, path: Path, save: bool = True) -> None:
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")


def run_pipeline(config_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
    config = load_config(config_path)
    paths = ensure_output_dirs(config)
    save_csv = bool(config["reports"].get("save_csv", True))
    save_figures = bool(config["reports"].get("save_figures", True))

    print("[1/10] Loading data...")
    data = load_market_data(config)

    print("[2/10] Building factor panel...")
    panel, factor_cols, factor_categories = build_factor_panel(data.prices, data.fundamentals, data.membership, config)
    rebal_dates = get_rebalance_dates(
        data.prices,
        start_date=config["backtest"]["start_date"],
        end_date=config["backtest"]["end_date"],
        freq=config["backtest"].get("rebalance_freq", "M"),
    )

    print("[3/10] Preprocessing factors and adding forward returns...")
    panel = preprocess_factor_panel(panel, factor_cols, config)
    panel = add_forward_returns(panel, data.prices, rebal_dates)
    panel = add_category_scores(
        panel,
        factor_categories,
        category_weights=config["factor"].get("category_weights"),
        score_col="score_equal",
    )
    panel = add_rolling_ic_weighted_score(
        panel,
        factor_cols=factor_cols,
        score_col="score_ic_weighted",
        lookback=int(config["factor"].get("ic_lookback", 12)),
        min_obs=int(config["factor"].get("min_ic_obs", 6)),
    )
    _save_csv(panel, paths["data"] / "factor_panel.csv", save_csv)

    print("[4/10] Running factor tests...")
    ic_df = calculate_ic(panel, factor_cols)
    ic_summary = summarize_ic(ic_df)
    corr = factor_correlation(panel, factor_cols)
    decay = rolling_factor_decay(ic_df, window=int(config["factor"].get("ic_lookback", 12)))
    quantiles = pd.concat(
        [
            quantile_analysis(panel, "score_equal", n_quantiles=int(config["factor"].get("quantiles", 5))),
            quantile_analysis(panel, "score_ic_weighted", n_quantiles=int(config["factor"].get("quantiles", 5))),
        ],
        ignore_index=True,
    )
    _save_csv(ic_df, paths["data"] / "ic_timeseries.csv", save_csv)
    _save_csv(ic_summary, paths["data"] / "ic_summary.csv", save_csv)
    _save_csv(corr.reset_index().rename(columns={"index": "factor"}), paths["data"] / "factor_correlation.csv", save_csv)
    _save_csv(decay, paths["data"] / "factor_decay.csv", save_csv)
    _save_csv(quantiles, paths["data"] / "quantile_returns.csv", save_csv)

    print("[5/10] Building walk-forward ML alpha and factor timing scores...")
    ml_diagnostics = pd.DataFrame()
    ml_importance = pd.DataFrame()
    ic_forecast_diagnostics = pd.DataFrame()
    factor_timing_weights = pd.DataFrame()
    score_cols = ["score_equal", "score_ic_weighted"]
    if bool(config.get("ml", {}).get("enabled", False)):
        panel, ml_diagnostics, ml_importance = add_walk_forward_ml_alpha_score(panel, factor_cols, config)
        score_cols.append("score_ml_alpha")
        panel, ic_forecast_diagnostics, factor_timing_weights = add_factor_ic_forecast_score(panel, ic_df, factor_cols, config)
        if "score_ic_forecast" in panel.columns and panel["score_ic_forecast"].notna().any():
            score_cols.append("score_ic_forecast")
        _save_csv(panel, paths["data"] / "factor_panel_with_ml.csv", save_csv)
        _save_csv(ml_diagnostics, paths["data"] / "ml_diagnostics.csv", save_csv)
        _save_csv(ml_importance, paths["data"] / "ml_feature_importance.csv", save_csv)
        _save_csv(ic_forecast_diagnostics, paths["data"] / "ic_forecast_diagnostics.csv", save_csv)
        _save_csv(factor_timing_weights, paths["data"] / "factor_timing_weights.csv", save_csv)

    print("[6/10] Building portfolios and running backtests...")
    all_perf = []
    all_yearly = []
    all_regime = []
    figure_paths: list[str] = []
    backtest_results: dict[str, pd.DataFrame] = {}
    weight_results: dict[str, pd.DataFrame] = {}

    regime_cfg = config["analysis"].get("regime", {})
    regime = classify_market_regime(
        data.benchmark,
        lookback_days=int(regime_cfg.get("lookback_days", 120)),
        bull_threshold=float(regime_cfg.get("bull_threshold", 0.10)),
        bear_threshold=float(regime_cfg.get("bear_threshold", -0.10)),
    )
    _save_csv(regime, paths["data"] / "market_regime.csv", save_csv)

    for score_col in score_cols:
        weights = build_target_weights(
            panel,
            score_col=score_col,
            top_n=config["backtest"].get("top_n"),
            top_pct=config["backtest"].get("top_pct"),
            industry_neutral=bool(config["backtest"].get("industry_neutral_selection", True)),
        )
        bt = run_backtest(
            data.prices,
            data.benchmark,
            weights,
            transaction_cost_bps=float(config["backtest"].get("transaction_cost_bps", 10)),
            execution_lag=int(config["backtest"].get("execution_lag", 1)),
        )
        perf = summarize_performance(bt)
        perf.insert(0, "score", score_col)
        y = yearly_returns(bt)
        y.insert(0, "score", score_col)
        r = summarize_by_regime(bt, regime)
        r.insert(0, "score", score_col)

        exposure = calculate_industry_exposure(weights)
        style_exposure = calculate_style_exposure(weights, panel, factor_cols)

        _save_csv(weights, paths["data"] / f"weights_{score_col}.csv", save_csv)
        _save_csv(bt, paths["data"] / f"backtest_{score_col}.csv", save_csv)
        _save_csv(y, paths["data"] / f"yearly_returns_{score_col}.csv", save_csv)
        _save_csv(exposure, paths["data"] / f"industry_exposure_{score_col}.csv", save_csv)
        _save_csv(style_exposure, paths["data"] / f"style_exposure_{score_col}.csv", save_csv)

        if save_figures:
            nav_path = paths["figures"] / f"nav_{score_col}.png"
            dd_path = paths["figures"] / f"drawdown_{score_col}.png"
            quantile_path = paths["figures"] / f"quantile_{score_col}.png"
            heatmap_path = paths["figures"] / f"monthly_heatmap_{score_col}.png"
            exposure_path = paths["figures"] / f"industry_exposure_{score_col}.png"
            plot_nav(bt, nav_path, title=f"NAV: {score_col}")
            plot_drawdown(bt, dd_path, title=f"Drawdown: {score_col}")
            plot_quantile_returns(quantiles, score_col, quantile_path)
            plot_monthly_heatmap(monthly_return_table(bt), heatmap_path, title=f"Monthly Returns: {score_col}")
            plot_industry_exposure(exposure, exposure_path)
            figure_paths.extend([str(nav_path), str(dd_path), str(quantile_path), str(heatmap_path), str(exposure_path)])

        all_perf.append(perf)
        all_yearly.append(y)
        all_regime.append(r)
        backtest_results[score_col] = bt
        weight_results[score_col] = weights

    perf_summary = pd.concat(all_perf, ignore_index=True)
    yearly = pd.concat(all_yearly, ignore_index=True)
    regime_summary = pd.concat(all_regime, ignore_index=True)
    _save_csv(perf_summary, paths["data"] / "performance_summary.csv", save_csv)
    _save_csv(yearly, paths["data"] / "yearly_returns.csv", save_csv)
    _save_csv(regime_summary, paths["data"] / "regime_summary.csv", save_csv)

    print("[7/10] Running sensitivity analysis...")
    sens_cfg = config["analysis"].get("sensitivity", {})
    sensitivity_parts = []
    for score_col in score_cols:
        sensitivity_parts.append(
            run_sensitivity_grid(
                data.prices,
                data.benchmark,
                panel,
                score_col=score_col,
                build_weights_func=build_target_weights,
                performance_func=summarize_performance,
                top_n_list=list(sens_cfg.get("top_n_list", [20, 30, 50])),
                cost_bps_list=list(sens_cfg.get("cost_bps_list", [5, 10, 20])),
                execution_lag=int(config["backtest"].get("execution_lag", 1)),
                industry_neutral=bool(config["backtest"].get("industry_neutral_selection", True)),
            )
        )
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True)
    _save_csv(sensitivity, paths["data"] / "sensitivity_summary.csv", save_csv)

    print("[8/10] Saving factor and ML figures...")
    if save_figures:
        corr_path = paths["figures"] / "factor_correlation.png"
        plot_factor_correlation(corr, corr_path)
        figure_paths.append(str(corr_path))
        top_factors = ic_summary[ic_summary["metric"] == "rank_ic"].sort_values("icir", ascending=False)["factor"].head(5)
        for factor in top_factors:
            path = paths["figures"] / f"rank_ic_{factor}.png"
            plot_ic(ic_df, factor, path)
            figure_paths.append(str(path))
        if bool(config.get("ml", {}).get("enabled", False)):
            imp_path = paths["figures"] / "ml_feature_importance.png"
            timing_path = paths["figures"] / "factor_timing_weights.png"
            plot_feature_importance(ml_importance, imp_path)
            plot_factor_timing_weights(factor_timing_weights, timing_path)
            if imp_path.exists():
                figure_paths.append(str(imp_path))
            if timing_path.exists():
                figure_paths.append(str(timing_path))

    print("[9/10] Writing markdown report...")
    write_markdown_report(
        paths["output"] / "report.md",
        config=config,
        performance=perf_summary,
        ic_summary=ic_summary,
        regime_summary=regime_summary,
        sensitivity=sensitivity,
        figure_paths=figure_paths,
        ml_diagnostics=ml_diagnostics,
        ml_importance=ml_importance,
        factor_timing_weights=factor_timing_weights,
    )

    print(f"[10/10] Done. Results saved to: {paths['output']}")
    return {
        "panel": panel,
        "ic": ic_df,
        "ic_summary": ic_summary,
        "performance": perf_summary,
        "regime": regime_summary,
        "sensitivity": sensitivity,
        "ml_diagnostics": ml_diagnostics,
        "ml_importance": ml_importance,
        "factor_timing_weights": factor_timing_weights,
        **{f"backtest_{k}": v for k, v in backtest_results.items()},
        **{f"weights_{k}": v for k, v in weight_results.items()},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run A-share multi-factor strategy backtest.")
    parser.add_argument("--config", type=str, default="configs/config_demo.yaml", help="Path to YAML config file.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
