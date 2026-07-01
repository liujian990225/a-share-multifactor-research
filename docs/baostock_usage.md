# BaoStock 免费数据源使用说明

本项目现在支持 BaoStock 作为免费 A 股真实数据源。BaoStock 不需要 token，适合在没有 Tushare/Wind/聚宽权限时跑通真实数据版多因子研究流程。

## 1. 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

如果你只是在旧环境中补装 BaoStock：

```bash
pip install baostock
```

## 2. 下载 BaoStock 数据

```bash
python scripts/fetch_baostock_data.py --config configs/config_baostock.yaml
```

脚本会生成项目标准 CSV：

```text
data/raw/baostock/prices.csv
data/raw/baostock/fundamentals.csv
data/raw/baostock/benchmark.csv
data/raw/baostock/membership.csv
data/raw/baostock/stock_basic.csv
```

## 3. 运行回测

```bash
python scripts/run_baostock_backtest.py
```

或者：

```bash
python -m mf_strategy.cli --config configs/config_baostock.yaml
```

输出目录：

```text
reports/baostock_run/
```

## 4. BaoStock 版本包含哪些因子？

BaoStock 日频行情可直接支持：

- 价值因子：PE_TTM、PB、PS_TTM 的倒数
- 动量因子：20/60/120 日收益率
- 低波动因子：20/60 日波动率、60 日最大回撤
- 流动性因子：成交额、换手率、Amihud 非流动性

免费日频数据不完整覆盖项目原本的质量因子，如 ROE、ROA、毛利率、净利率、资产负债率。为了保证流程可运行，脚本会在 `fundamentals.csv` 中写入中性默认值。后续如果你接入 Wind、聚宽、Tushare 高权限或自己整理财报数据，可以替换这些字段，质量因子会自动参与计算。

## 5. 第一次运行建议

默认配置只下载 80 只股票，回测区间为 2021-2024，目的是让新手先快速跑通完整流程。跑通后你可以修改：

```yaml
data:
  baostock:
    start_date: "2017-01-01"
    end_date: "2024-12-31"
    max_symbols: 300

backtest:
  start_date: "2018-01-01"
  end_date: "2024-12-31"
  top_n: 30
```

如果运行较慢，可以先关闭机器学习模块：

```yaml
ml:
  enabled: false
```

## 6. 上传 GitHub 注意事项

不要上传真实数据和报告输出。`.gitignore` 已默认忽略：

```text
data/
reports/
*.csv
*.parquet
*.pkl
```

GitHub 上保留代码、配置和文档即可。面试时可以在本地展示 `reports/baostock_run/report.md` 和图表。
