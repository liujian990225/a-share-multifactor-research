from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _zscore_by_date(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _rank_to_score(s: pd.Series) -> pd.Series:
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=s.index)
    pct = s.rank(pct=True, method="average")
    return (pct - 0.5) * 2.0


def _make_regressor(model_type: str, random_state: int = 42, n_estimators: int = 200, max_depth: int = 3):
    """Create a tree model with graceful fallback.

    xgboost is optional because many learners cannot install it quickly on Windows.
    If model_type='xgboost' but xgboost is unavailable, the function falls back to
    sklearn GradientBoostingRegressor and records the effective model name outside.
    """
    model_type = (model_type or "random_forest").lower()
    if model_type in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBRegressor  # type: ignore

            return XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=1,
            ), "xgboost"
        except Exception:
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor(
                n_estimators=max(80, n_estimators // 2),
                max_depth=max_depth,
                learning_rate=0.05,
                random_state=random_state,
            ), "gradient_boosting_fallback"

    if model_type in {"gradient_boosting", "gbrt"}:
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            random_state=random_state,
        ), "gradient_boosting"

    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    ), "random_forest"


def _safe_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return pd.DataFrame({"feature": feature_cols, "importance": np.nan})
    out = pd.DataFrame({"feature": feature_cols, "importance": np.asarray(values, dtype=float)})
    total = out["importance"].sum()
    if total > 0:
        out["importance"] = out["importance"] / total
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def add_walk_forward_ml_alpha_score(
    panel: pd.DataFrame,
    factor_cols: list[str],
    config: Mapping[str, Any],
    score_col: str = "score_ml_alpha",
    fwd_return_col: str = "fwd_return",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add a walk-forward ML alpha score to the factor panel.

    At each rebalance date, the model trains only on earlier dates and predicts the
    current cross-section. This avoids look-ahead bias. The target is next-period
    forward return, winsorized cross-sectionally by date and converted to ranks.
    """
    ml_cfg = config.get("ml", {})
    if not bool(ml_cfg.get("enabled", False)):
        out = panel.copy()
        out[score_col] = np.nan
        return out, pd.DataFrame(), pd.DataFrame()

    try:
        import sklearn  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError("ML mode requires scikit-learn. Install requirements.txt first.") from exc

    model_type = str(ml_cfg.get("model_type", "random_forest"))
    random_state = int(config.get("project", {}).get("random_seed", 42))
    n_estimators = int(ml_cfg.get("n_estimators", 200))
    max_depth = int(ml_cfg.get("max_depth", 3))
    train_window = int(ml_cfg.get("train_window_months", 36))
    min_train_dates = int(ml_cfg.get("min_train_months", 18))
    validation_months = int(ml_cfg.get("validation_months", 6))
    use_rank_target = bool(ml_cfg.get("rank_target", True))

    feature_cols = [c for c in factor_cols + ["size_log_mcap"] if c in panel.columns]
    out = panel.copy().sort_values(["date", "symbol"]).reset_index(drop=True)
    out[score_col] = np.nan

    work = out[["date", "symbol", fwd_return_col] + feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    if use_rank_target:
        work["target"] = work.groupby("date")[fwd_return_col].transform(_rank_to_score)
    else:
        work["target"] = work.groupby("date")[fwd_return_col].transform(lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))
    work = work.dropna(subset=feature_cols + ["target"])

    dates = sorted(out["date"].dropna().unique())
    diagnostics: list[dict[str, Any]] = []
    importances: list[pd.DataFrame] = []

    for date in dates:
        date_ts = pd.Timestamp(date)
        past_dates = [pd.Timestamp(d) for d in dates if pd.Timestamp(d) < date_ts]
        if len(past_dates) < min_train_dates:
            continue
        train_dates = past_dates[-train_window:]
        train = work[work["date"].isin(train_dates)].copy()
        pred_mask = out["date"] == date_ts
        pred = out.loc[pred_mask, ["date", "symbol"] + feature_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
        if len(train) < 200 or pred.empty:
            continue

        model, effective_model = _make_regressor(model_type, random_state, n_estimators, max_depth)
        X_train = train[feature_cols]
        y_train = train["target"]
        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(pred[feature_cols]), index=pred.index)
        out.loc[y_pred.index, score_col] = y_pred

        val_ic = np.nan
        val_rank_ic = np.nan
        val_dates = train_dates[-validation_months:] if validation_months > 0 else []
        if val_dates:
            train_sub = train[~train["date"].isin(val_dates)]
            val_sub = train[train["date"].isin(val_dates)]
            if len(train_sub) >= 100 and len(val_sub) >= 30:
                val_model, _ = _make_regressor(model_type, random_state, n_estimators, max_depth)
                val_model.fit(train_sub[feature_cols], train_sub["target"])
                val_pred = pd.Series(val_model.predict(val_sub[feature_cols]), index=val_sub.index)
                if val_pred.nunique() > 1 and val_sub["target"].nunique() > 1:
                    val_ic = val_pred.corr(val_sub["target"], method="pearson")
                    val_rank_ic = val_pred.corr(val_sub["target"], method="spearman")

        diagnostics.append(
            {
                "date": date_ts,
                "score": score_col,
                "model": effective_model,
                "train_start": min(train_dates),
                "train_end": max(train_dates),
                "train_rows": int(len(train)),
                "n_features": int(len(feature_cols)),
                "prediction_rows": int(len(pred)),
                "validation_ic": val_ic,
                "validation_rank_ic": val_rank_ic,
            }
        )
        imp = _safe_feature_importance(model, feature_cols)
        imp.insert(0, "date", date_ts)
        imp.insert(1, "score", score_col)
        importances.append(imp)

    out[score_col] = out.groupby("date")[score_col].transform(_zscore_by_date)
    diagnostics_df = pd.DataFrame(diagnostics)
    importances_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    return out, diagnostics_df, importances_df


def _ic_feature_frame(ic_df: pd.DataFrame, factor_cols: list[str], lookback: int) -> pd.DataFrame:
    """Create factor-date features from past Rank IC only."""
    records: list[dict[str, Any]] = []
    wide = ic_df.pivot(index="date", columns="factor", values="rank_ic").sort_index().reindex(columns=factor_cols)
    for factor in factor_cols:
        s = wide[factor]
        for date in wide.index:
            hist = s.loc[:date].iloc[:-1].dropna().tail(lookback)
            if hist.empty:
                continue
            records.append(
                {
                    "date": pd.Timestamp(date),
                    "factor": factor,
                    "ic_last": hist.iloc[-1],
                    "ic_mean": hist.mean(),
                    "ic_std": hist.std(ddof=1) if len(hist) > 1 else 0.0,
                    "ic_win_rate": (hist > 0).mean(),
                    "ic_tstat": hist.mean() / (hist.std(ddof=1) / np.sqrt(len(hist))) if len(hist) > 1 and hist.std(ddof=1) > 0 else 0.0,
                    "ic_obs": len(hist),
                    "target_next_ic": s.loc[date],
                }
            )
    return pd.DataFrame(records)


def add_factor_ic_forecast_score(
    panel: pd.DataFrame,
    ic_df: pd.DataFrame,
    factor_cols: list[str],
    config: Mapping[str, Any],
    score_col: str = "score_ic_forecast",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Forecast factor IC and turn predicted IC into dynamic factor weights.

    This is a factor-timing layer: at each date it predicts which factors are more
    likely to work next period, then uses positive predicted IC values as weights.
    """
    ml_cfg = config.get("ml", {})
    if not bool(ml_cfg.get("enabled", False)) or not bool(ml_cfg.get("factor_ic_forecast", True)):
        out = panel.copy()
        out[score_col] = np.nan
        return out, pd.DataFrame(), pd.DataFrame()

    model_type = str(ml_cfg.get("ic_model_type", ml_cfg.get("model_type", "random_forest")))
    random_state = int(config.get("project", {}).get("random_seed", 42))
    lookback = int(ml_cfg.get("ic_feature_lookback", config.get("factor", {}).get("ic_lookback", 12)))
    min_train_rows = int(ml_cfg.get("ic_min_train_rows", 80))
    n_estimators = int(ml_cfg.get("n_estimators", 200))
    max_depth = int(ml_cfg.get("max_depth", 3))
    feature_cols = ["ic_last", "ic_mean", "ic_std", "ic_win_rate", "ic_tstat", "ic_obs"]

    out = panel.copy().sort_values(["date", "symbol"]).reset_index(drop=True)
    out[score_col] = np.nan
    feat = _ic_feature_frame(ic_df, factor_cols, lookback).replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    dates = sorted(out["date"].dropna().unique())
    weight_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    equal = pd.Series(1 / len(factor_cols), index=factor_cols, dtype=float)
    for date in dates:
        date_ts = pd.Timestamp(date)
        train = feat[(feat["date"] < date_ts) & feat["target_next_ic"].notna()].dropna(subset=["target_next_ic"])
        current = feat[feat["date"] == date_ts]
        if len(train) < min_train_rows or current.empty:
            weights = equal.copy()
            effective_model = "equal_fallback"
        else:
            model, effective_model = _make_regressor(model_type, random_state, n_estimators, max_depth)
            model.fit(train[feature_cols], train["target_next_ic"])
            current = current.copy()
            current["predicted_ic"] = model.predict(current[feature_cols])
            pred = current.set_index("factor")["predicted_ic"].reindex(factor_cols).fillna(0.0)
            weights = pred.clip(lower=0.0)
            if weights.sum() <= 0:
                weights = equal.copy()
            else:
                weights = weights / weights.sum()

        mask = out["date"] == date_ts
        out.loc[mask, score_col] = out.loc[mask, factor_cols].mul(weights, axis=1).sum(axis=1).values
        for factor, w in weights.items():
            weight_rows.append({"date": date_ts, "factor": factor, "predicted_weight": float(w)})
        diagnostics.append({"date": date_ts, "score": score_col, "model": effective_model, "train_rows": int(len(train))})

    out[score_col] = out.groupby("date")[score_col].transform(_zscore_by_date)
    return out, pd.DataFrame(diagnostics), pd.DataFrame(weight_rows)
