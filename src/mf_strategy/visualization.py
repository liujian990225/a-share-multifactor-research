from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _prepare_path(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_nav(backtest: pd.DataFrame, path: str | Path, title: str = "Net Asset Value") -> None:
    path = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(backtest["date"], backtest["nav"], label="Strategy")
    ax.plot(backtest["date"], backtest["benchmark_nav"], label="Benchmark")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drawdown(backtest: pd.DataFrame, path: str | Path, title: str = "Drawdown") -> None:
    path = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(backtest["date"], backtest["drawdown"], label="Strategy")
    ax.plot(backtest["date"], backtest["benchmark_drawdown"], label="Benchmark")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ic(ic_df: pd.DataFrame, factor: str, path: str | Path) -> None:
    path = _prepare_path(path)
    sub = ic_df[ic_df["factor"] == factor].sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(sub["date"], sub["rank_ic"], width=20, label="Rank IC")
    ax.plot(sub["date"], sub["rank_ic"].rolling(12, min_periods=3).mean(), label="12M Rolling Mean")
    ax.axhline(0, linewidth=1)
    ax.set_title(f"Rank IC: {factor}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rank IC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_quantile_returns(quantile_df: pd.DataFrame, score_col: str, path: str | Path) -> None:
    path = _prepare_path(path)
    sub = quantile_df[quantile_df["score"] == score_col].sort_values("date")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    for quantile, group in sub.groupby("quantile"):
        ax.plot(group["date"], 1 + group["cum_return"], label=str(quantile))
    ax.set_title(f"Quantile Cumulative Return: {score_col}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative NAV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_factor_correlation(corr: pd.DataFrame, path: str | Path) -> None:
    path = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.to_numpy(), aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Average Cross-sectional Factor Correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_monthly_heatmap(monthly_table: pd.DataFrame, path: str | Path, title: str = "Monthly Returns") -> None:
    path = _prepare_path(path)
    if monthly_table.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(monthly_table))))
    data = monthly_table.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([str(i) for i in range(1, 13)])
    ax.set_yticks(np.arange(len(monthly_table.index)))
    ax.set_yticklabels(monthly_table.index.astype(str))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1%}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_industry_exposure(exposure: pd.DataFrame, path: str | Path) -> None:
    path = _prepare_path(path)
    if exposure.empty:
        return
    pivot = exposure.pivot(index="date", columns="industry", values="industry_weight").fillna(0.0)
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot.area(ax=ax, stacked=True, linewidth=0)
    ax.set_title("Portfolio Industry Exposure")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_feature_importance(importance: pd.DataFrame, path: str | Path, top_n: int = 15) -> None:
    path = _prepare_path(path)
    if importance.empty or "feature" not in importance.columns:
        return
    if "importance" not in importance.columns:
        return
    imp = importance.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(top_n)
    if imp.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(imp))))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1])
    ax.set_title("Average ML Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_factor_timing_weights(weights: pd.DataFrame, path: str | Path) -> None:
    path = _prepare_path(path)
    if weights.empty or not {"date", "factor", "predicted_weight"}.issubset(weights.columns):
        return
    pivot = weights.pivot_table(index="date", columns="factor", values="predicted_weight", aggfunc="mean").fillna(0.0)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot.area(ax=ax, stacked=True, linewidth=0)
    ax.set_title("Predicted Factor Timing Weights")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
