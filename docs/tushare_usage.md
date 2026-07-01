# Tushare 真实数据接入指南

本项目已经内置 Tushare 数据下载脚本。推荐采用“两步式”流程：

1. 先从 Tushare 下载真实数据，并保存为项目标准 CSV。
2. 再从本地 CSV 运行回测，避免每次调参都重复请求接口。

## 1. 准备 Tushare Token

注册并登录 Tushare Pro 后获取 token。不要把 token 写入 GitHub。

Windows PowerShell：

```powershell
setx TUSHARE_TOKEN "你的token"
```

关闭并重新打开终端后生效。

macOS / Linux：

```bash
export TUSHARE_TOKEN="你的token"
```

如果想长期生效，可以把上一行加入 `~/.zshrc` 或 `~/.bashrc`。

## 2. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## 3. 下载真实数据

默认配置文件是：

```text
configs/config_tushare.yaml
```

默认设置：

- 股票池：沪深 300 历史成分股，`index_code=000300.SH`
- 基准：沪深 300 指数，`benchmark_index=000300.SH`
- 时间：2017-01-01 至 2024-12-31
- 复权：前复权，`adj=qfq`
- 输出目录：`data/raw/tushare/`

运行：

```bash
python scripts/fetch_tushare_data.py --config configs/config_tushare.yaml
```

如果你的 Tushare 权限不足以调用 `index_weight`，脚本会自动回退为 `stock_basic` 中的前若干只股票。你也可以在配置中改成 `custom` 模式：

```yaml
data:
  tushare:
    universe:
      mode: custom
      symbols: ["600519.SH", "000001.SZ", "300750.SZ"]
```

如果要强制重新下载，使用：

```bash
python scripts/fetch_tushare_data.py --config configs/config_tushare.yaml --force
```

## 4. 运行真实数据回测

下载完成后，运行：

```bash
python -m mf_strategy.cli --config configs/config_tushare.yaml
```

或者：

```bash
python scripts/run_tushare_backtest.py
```

输出结果在：

```text
reports/tushare_run/
├── data/
│   ├── factor_panel.csv
│   ├── ic_summary.csv
│   ├── performance_summary.csv
│   └── ...
├── figures/
│   ├── nav_score_equal.png
│   ├── nav_score_ic_weighted.png
│   ├── drawdown_score_equal.png
│   └── ...
└── report.md
```

## 5. 数据字段映射

脚本会把 Tushare 原始字段转换成项目标准字段。

### prices.csv

| 项目字段 | Tushare 来源 | 说明 |
|---|---|---|
| date | trade_date | 交易日 |
| symbol | ts_code | 股票代码 |
| open/high/low/close | pro_bar 或 daily | 前复权价格，取决于 `use_pro_bar` 和 `adj` 设置 |
| volume | vol | 成交量 |
| amount | amount * 1000 | Tushare 通常为千元，项目转换为元 |
| turnover_rate | daily_basic.turnover_rate | 换手率 |
| listing_date | stock_basic.list_date | 上市日期 |
| is_tradable | 固定为 1 | 简化可交易标记 |
| is_st | stock_basic.name 判断 | 当前名称中含 ST 或退则记为 1 |

### fundamentals.csv

| 项目字段 | Tushare 来源 | 说明 |
|---|---|---|
| pe_ttm | daily_basic.pe_ttm | TTM 市盈率 |
| pb | daily_basic.pb | 市净率 |
| ps_ttm | daily_basic.ps_ttm | TTM 市销率 |
| dividend_yield | daily_basic.dv_ttm | TTM 股息率 |
| market_cap | daily_basic.total_mv * 10000 | 总市值，单位转为元 |
| roe | fina_indicator.roe | ROE |
| roa | fina_indicator.roa | ROA |
| gross_margin | fina_indicator.grossprofit_margin | 毛利率 |
| net_margin | fina_indicator.netprofit_margin | 净利率 |
| debt_to_asset | fina_indicator.debt_to_assets | 资产负债率 |
| industry | stock_basic.industry | 行业 |

质量类财务指标使用 `ann_date` 对齐到调仓日前最近一次已公告数据，尽量避免未来函数。如果 `ann_date` 缺失，则使用 `end_date + quality_lag_days` 的保守近似。

## 6. 常见问题

### 1. 下载很慢怎么办？

沪深 300 多年数据需要请求较多接口。第一次下载慢是正常的。脚本会缓存每只股票的原始数据到：

```text
data/raw/tushare/raw_by_symbol/
```

后续重复运行会优先读缓存。

### 2. 权限不足怎么办？

Tushare 不同接口有积分权限要求。你可以先把 `universe.mode` 改成 `custom`，用 10 到 30 只股票跑通流程，再扩大股票池。

### 3. GitHub 要不要上传数据？

不建议上传真实行情和财务 CSV。项目 `.gitignore` 已忽略 `data/raw/*.csv` 和 `reports/*`。建议只上传代码、配置模板和报告截图，或者上传小规模示例数据。

### 4. 为什么仍然 `data.source: csv`？

这是为了工程稳定性。真实数据先下载成标准 CSV，回测只依赖本地文件。这样调参、画图、生成报告时不会反复请求 Tushare。
