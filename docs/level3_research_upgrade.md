# Level 3 量化研究员级升级说明

本版本在原有 Tushare 真实数据多因子回测框架上增加了研究员级模块，重点不是简单提高历史收益，而是增强“可研究、可解释、可验证、可答辩”的能力。

## 新增能力

### 1. Walk-forward 机器学习 Alpha 模型

新增 `src/mf_strategy/ml_alpha.py`，在每个调仓日使用历史截面样本训练模型，并预测当前截面的股票 Alpha 得分。

默认流程：

1. 使用价值、质量、动量、低波动、流动性、市值等因子作为特征；
2. 使用下一期收益的截面排名作为训练目标；
3. 每个调仓日只使用此前月份训练，避免未来函数；
4. 输出 `score_ml_alpha`；
5. 对 `score_ml_alpha` 单独构建组合并回测。

默认模型为 `random_forest`，也支持：

```yaml
ml:
  model_type: random_forest      # random_forest | gradient_boosting | xgboost
```

如果配置为 `xgboost` 但本地未安装 `xgboost`，代码会自动 fallback 到 `GradientBoostingRegressor`。

### 2. 因子 IC 预测 / 因子择时

新增 `score_ic_forecast`：

1. 对每个因子构造历史 Rank IC 特征；
2. 训练模型预测下一期因子 IC；
3. 使用预测 IC 的正值归一化为动态因子权重；
4. 用动态权重合成综合分。

输出文件：

```text
data/ic_forecast_diagnostics.csv
data/factor_timing_weights.csv
figures/factor_timing_weights.png
```

### 3. 机器学习解释性输出

输出：

```text
data/ml_diagnostics.csv
data/ml_feature_importance.csv
figures/ml_feature_importance.png
```

这些文件用于回答面试问题：

- 模型用了哪些特征？
- 哪些因子贡献更高？
- 模型是否严格使用历史训练？
- 样本外验证 Rank IC 怎么样？

### 4. 多策略对比

原来回测两组策略：

```text
score_equal
score_ic_weighted
```

Level 3 新增：

```text
score_ml_alpha
score_ic_forecast
```

最终会在 `reports/tushare_run/data/performance_summary.csv` 中比较四类策略。

## 推荐运行方式

先下载 Tushare 数据：

```bash
python scripts/fetch_tushare_data.py --config configs/config_tushare.yaml
```

运行 Level 3 回测：

```bash
python -m mf_strategy.cli --config configs/config_tushare.yaml
```

如果机器性能较弱，可以临时降低模型复杂度：

```yaml
ml:
  enabled: true
  model_type: random_forest
  n_estimators: 60
  max_depth: 3
  train_window_months: 24
```

如果只想跑基础版本：

```yaml
ml:
  enabled: false
```

## 面试表达建议

可以这样介绍升级点：

> 我在基础多因子框架上进一步加入了 walk-forward 机器学习 Alpha 模型和因子 IC 预测模块。ML Alpha 模型用历史截面的因子特征预测下一期收益排名，因子 IC 预测模块则基于各因子的历史 Rank IC 统计特征动态调整因子权重。整个过程严格按时间滚动训练，当前调仓日只使用历史样本，避免未来函数。同时输出特征重要性、模型验证 Rank IC 和因子择时权重，用于解释模型有效性。

## 注意事项

- 机器学习模块会增加运行时间；真实沪深 300 长周期数据建议本地运行。
- 不建议直接把 `data/raw/tushare/` 和 `reports/` 上传到 GitHub，项目 `.gitignore` 已默认忽略。
- 机器学习结果不一定优于等权多因子，真实项目中应重视样本外稳定性，而不是只追求历史最优收益。
