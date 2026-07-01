from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.2%}"


def _fmt_num(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.3f}"


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data._"
    return df.head(max_rows).to_markdown(index=False)


def write_markdown_report(
    output_path: str | Path,
    config: dict,
    performance: pd.DataFrame,
    ic_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    figure_paths: list[str],
    ml_diagnostics: pd.DataFrame | None = None,
    ml_importance: pd.DataFrame | None = None,
    factor_timing_weights: pd.DataFrame | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    perf_display = performance.copy()
    percent_cols = [c for c in perf_display.columns if "return" in c or "drawdown" in c or "volatility" in c or "rate" in c]
    for col in percent_cols:
        if col in perf_display:
            perf_display[col] = perf_display[col].map(_fmt_pct)
    for col in ["sharpe", "sortino", "calmar", "information_ratio", "beta", "alpha"]:
        if col in perf_display:
            perf_display[col] = perf_display[col].map(_fmt_num)

    lines = [
        "# 多因子选股策略回测报告",
        "",
        "## 1. 项目设置",
        "",
        f"- 数据源：`{config['data']['source']}`",
        f"- 回测区间：`{config['backtest']['start_date']}` 到 `{config['backtest']['end_date']}`",
        f"- 调仓频率：`{config['backtest']['rebalance_freq']}`",
        f"- Top N：`{config['backtest']['top_n']}`",
        f"- 单边交易成本：`{config['backtest']['transaction_cost_bps']} bps`",
        f"- 行业中性选股：`{config['backtest']['industry_neutral_selection']}`",
        "",
        "## 2. 绩效摘要",
        "",
        dataframe_to_markdown(perf_display),
        "",
        "## 3. IC 摘要，按 ICIR 排序",
        "",
        dataframe_to_markdown(ic_summary.sort_values('icir', ascending=False), max_rows=15),
        "",
        "## 4. 市场环境分阶段表现",
        "",
        dataframe_to_markdown(regime_summary),
        "",
        "## 5. 参数敏感性，节选",
        "",
        dataframe_to_markdown(sensitivity, max_rows=15),
        "",
    ]

    if bool(config.get("ml", {}).get("enabled", False)):
        lines.extend([
            "## 6. Level 3：机器学习 Alpha 与因子择时",
            "",
            "本节使用 walk-forward 方式训练模型：每个调仓日只使用历史样本训练，再预测当前截面的股票得分，避免未来函数。",
            "",
            "### 6.1 ML 训练诊断，节选",
            "",
            dataframe_to_markdown(ml_diagnostics if ml_diagnostics is not None else pd.DataFrame(), max_rows=12),
            "",
            "### 6.2 平均特征重要性，节选",
            "",
            dataframe_to_markdown(
                (ml_importance.groupby('feature', as_index=False)['importance'].mean().sort_values('importance', ascending=False)
                 if ml_importance is not None and not ml_importance.empty and 'importance' in ml_importance.columns else pd.DataFrame()),
                max_rows=15,
            ),
            "",
            "### 6.3 因子 IC 预测权重，节选",
            "",
            dataframe_to_markdown(factor_timing_weights if factor_timing_weights is not None else pd.DataFrame(), max_rows=15),
            "",
        ])

    lines.extend([
        "## 7. 图表索引",
        "",
    ])

    for fig in figure_paths:
        rel = Path(fig).as_posix()
        lines.append(f"- `{rel}`")

    lines.extend(
        [
            "",
            "## 8. 结论模板",
            "",
            "真实数据跑完后，可以在这里补充：",
            "",
            "1. 哪些因子 Rank IC 更稳定；",
            "2. 等权合成、滚动 IC 加权、ML Alpha 和因子 IC 预测权重哪个效果更好；",
            "3. 策略在哪类市场环境中表现较强或较弱；",
            "4. 行业暴露是否过于集中；",
            "5. 对交易成本和持仓数量是否敏感。",
            "",
            "> 注意：demo 数据只用于验证项目流程，不可作为真实投资结论。",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
