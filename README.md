# A 股多因子选股策略研究与回测系统

这是一个适合写进简历和 GitHub 的「第三阶段亮点版」量化项目，覆盖从数据读取、因子构建、因子检验、多因子合成、组合构建、回测评估、行业/风格暴露分析、市场环境分阶段分析到可视化报告的完整流程。

> 项目默认使用自动生成的 demo 数据跑通全流程；当前版本已接入 Tushare 下载脚本，可直接拉取真实 A 股数据并转换为标准 CSV 后运行回测。

## 项目亮点

- **完整多因子框架**：价值、质量、动量、低波动、流动性五大类因子。
- **标准因子工程**：缺失值填充、MAD 去极值、Z-score 标准化、行业与市值中性化。
- **因子有效性检验**：Pearson IC、Spearman Rank IC、ICIR、IC 胜率、分层回测、因子相关性。
- **多因子合成对比**：等权合成 vs 滚动 Rank IC 加权合成。
- **更真实的回测设定**：月度调仓、Top N 组合、行业中性选股、交易成本、换手率统计。
- **进阶归因分析**：牛市/熊市/震荡市表现、行业暴露、风格暴露、参数敏感性分析。
- **可视化报告**：净值曲线、回撤曲线、月度收益热力图、IC 时序图、分层收益图、因子相关性热力图。
- **Tushare 真实数据接入**：支持自动下载行情、估值、财务指标、指数基准和指数成分，并缓存为标准 CSV。

## 快速开始

```bash
# 1. 创建环境，Python 3.10+ 推荐
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt
pip install -e .

# 3. 运行 demo 全流程
python -m mf_strategy.cli --config configs/config_demo.yaml
```

运行完成后，结果会保存到：

```text
reports/demo_run/
├── data/
│   ├── factor_panel.csv
│   ├── ic_summary.csv
│   ├── performance_summary.csv
│   ├── regime_summary.csv
│   ├── sensitivity_summary.csv
│   └── ...
├── figures/
│   ├── nav_score_equal.png
│   ├── nav_score_ic_weighted.png
│   ├── drawdown_score_equal.png
│   ├── monthly_heatmap_score_ic_weighted.png
│   └── ...
└── report.md
```

## 使用 Tushare 真实数据

本项目推荐用 Tushare 两步式接入真实数据：先下载缓存，再本地回测。

```bash
# 1. 设置 token，Windows PowerShell 示例
setx TUSHARE_TOKEN "你的token"

# macOS / Linux 示例
export TUSHARE_TOKEN="你的token"

# 2. 下载并标准化 Tushare 数据
python scripts/fetch_tushare_data.py --config configs/config_tushare.yaml

# 3. 运行真实数据回测
python -m mf_strategy.cli --config configs/config_tushare.yaml
# 或者
python scripts/run_tushare_backtest.py
```

结果会输出到：

```text
reports/tushare_run/
```

Tushare 详细使用说明见：[`docs/tushare_usage.md`](docs/tushare_usage.md)。

## 使用自备 CSV 数据

将你的真实数据放到 `data/raw/`，至少需要以下 3 个文件：

```text
data/raw/prices.csv
data/raw/fundamentals.csv
data/raw/benchmark.csv
```

然后复制配置文件：

```bash
cp configs/config_csv.yaml.example configs/config_csv.yaml
```

修改其中的路径，再运行：

```bash
python -m mf_strategy.cli --config configs/config_csv.yaml
```

详细字段说明见：[`docs/data_schema.md`](docs/data_schema.md)。

## 项目结构

```text
a_share_multifactor_backtest/
├── configs/
│   ├── config_demo.yaml
│   ├── config_tushare.yaml
│   └── config_csv.yaml.example
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── methodology.md
│   ├── data_schema.md
│   ├── tushare_usage.md
│   └── resume_bullets.md
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_factor_analysis.ipynb
│   └── 03_backtest_result.ipynb
├── reports/
├── scripts/
│   ├── run_demo.py
│   ├── fetch_tushare_data.py
│   └── run_tushare_backtest.py
├── src/
│   └── mf_strategy/
│       ├── config.py
│       ├── data_loader.py
│       ├── demo_data.py
│       ├── tushare_loader.py
│       ├── factor_engine.py
│       ├── factor_preprocess.py
│       ├── factor_test.py
│       ├── portfolio.py
│       ├── backtest.py
│       ├── performance.py
│       ├── market_regime.py
│       ├── visualization.py
│       ├── report.py
│       └── cli.py
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 推荐简历写法

```text
A 股多因子选股策略研究与回测系统 | Python, Pandas, NumPy, Matplotlib
- 构建覆盖价值、质量、动量、低波动和流动性因子的 A 股多因子选股框架，完成因子计算、预处理、有效性检验、组合构建和回测评估全流程。
- 对截面因子进行缺失值填充、MAD 去极值、Z-score 标准化及行业/市值中性化处理，并使用 IC、Rank IC、ICIR、分层回测和因子相关性分析评估因子有效性。
- 对比等权合成与滚动 Rank IC 加权两种多因子模型，采用月度调仓、Top N 选股、行业中性约束和交易成本设定构建组合。
- 输出年化收益、夏普比率、最大回撤、信息比率、换手率、超额收益、市场环境分阶段表现和行业/风格暴露分析，形成可复现实证报告。
```

## 注意事项

1. demo 数据用于验证流程，不代表真实市场收益。
2. 使用真实 A 股数据时，财务因子建议按公告日期对齐，避免未来函数。
3. 回测默认使用收盘价和下一交易日执行，适合作为研究框架；实盘前应加入更精细的滑点、停牌、涨跌停和成交约束。

---

## Level 3：量化研究员级升级

本项目现已支持研究员级增强模块，可在 Tushare 真实数据上运行：

- `score_equal`：静态/等权多因子合成；
- `score_ic_weighted`：滚动 Rank IC 加权合成；
- `score_ml_alpha`：walk-forward 机器学习 Alpha 模型；
- `score_ic_forecast`：因子 Rank IC 预测与动态因子择时。

### 运行 Level 3 Tushare 版本

```bash
python scripts/fetch_tushare_data.py --config configs/config_tushare.yaml
python scripts/run_level3_tushare_backtest.py
```

或直接：

```bash
python -m mf_strategy.cli --config configs/config_tushare.yaml
```

### Level 3 输出文件

```text
reports/tushare_run/
├── report.md
├── data/
│   ├── performance_summary.csv
│   ├── factor_panel_with_ml.csv
│   ├── ml_diagnostics.csv
│   ├── ml_feature_importance.csv
│   ├── factor_timing_weights.csv
│   ├── backtest_score_ml_alpha.csv
│   └── backtest_score_ic_forecast.csv
└── figures/
    ├── nav_score_ml_alpha.png
    ├── nav_score_ic_forecast.png
    ├── ml_feature_importance.png
    └── factor_timing_weights.png
```

### 配置机器学习模块

在 `configs/config_tushare.yaml` 中修改：

```yaml
ml:
  enabled: true
  model_type: random_forest      # random_forest | gradient_boosting | xgboost
  n_estimators: 100
  max_depth: 3
  train_window_months: 36
  min_train_months: 18
  validation_months: 6
  factor_ic_forecast: true
```

若希望使用 XGBoost：

```bash
pip install -e ".[ml]"
```

然后修改：

```yaml
ml:
  model_type: xgboost
  ic_model_type: xgboost
```

如未安装 XGBoost，程序会自动 fallback 到 sklearn 的 Gradient Boosting 模型。

### 研究注意事项

机器学习模块严格采用 walk-forward 训练方式：当前调仓日的预测只使用历史训练样本，避免未来函数。真实研究中不要只比较最终收益，更要关注样本外 Rank IC、最大回撤、换手率、交易成本、行业暴露和不同市场环境下的稳定性。

更多说明见：`docs/level3_research_upgrade.md`。


## Free BaoStock Data Source

If your Tushare account does not have access to interfaces such as `index_weight`, `fina_indicator`, or `adj_factor`, use the BaoStock workflow. BaoStock is free and does not require a token, so it is suitable for running a real-data version of this project locally.

```bash
pip install -r requirements.txt
pip install -e .
python scripts/fetch_baostock_data.py --config configs/config_baostock.yaml
python scripts/run_baostock_backtest.py
```

The BaoStock script writes standard project CSV files to `data/raw/baostock/`, and the report is generated under `reports/baostock_run/`. The free daily BaoStock dataset supports value, momentum, low-volatility and liquidity factors directly. Quality factors are kept as neutral placeholders unless you later add complete point-in-time financial data.

See `docs/baostock_usage.md` for detailed instructions.
