# 真实数据字段说明

## 1. prices.csv

每日行情数据，一行代表一只股票一个交易日。

必需字段：

| 字段 | 类型 | 示例 | 说明 |
|---|---:|---|---|
| date | date | 2024-01-31 | 交易日 |
| symbol | str | 000001.SZ | 股票代码 |
| open | float | 10.50 | 开盘价 |
| high | float | 10.80 | 最高价 |
| low | float | 10.20 | 最低价 |
| close | float | 10.60 | 收盘价，建议使用复权价 |
| volume | float | 12000000 | 成交量 |
| amount | float | 130000000 | 成交额，单位保持一致即可 |

可选字段：

| 字段 | 类型 | 说明 |
|---|---:|---|
| turnover_rate | float | 换手率 |
| is_tradable | bool/int | 是否可交易，1 为可交易 |
| is_st | bool/int | 是否 ST，1 为 ST |
| listing_date | date | 上市日期 |

## 2. fundamentals.csv

财务和估值数据，一行代表一个股票在一个已知日期的基本面快照。

必需字段：

| 字段 | 类型 | 说明 |
|---|---:|---|
| date | date | 数据可用日期。真实回测中建议使用公告日期或数据发布日期，避免未来函数。 |
| symbol | str | 股票代码 |
| pe_ttm | float | TTM 市盈率 |
| pb | float | 市净率 |
| ps_ttm | float | TTM 市销率 |
| dividend_yield | float | 股息率 |
| roe | float | ROE |
| roa | float | ROA |
| gross_margin | float | 毛利率 |
| net_margin | float | 净利率 |
| debt_to_asset | float | 资产负债率 |
| market_cap | float | 总市值或流通市值 |
| industry | str | 行业名称 |

## 3. benchmark.csv

基准指数数据。

必需字段：

| 字段 | 类型 | 说明 |
|---|---:|---|
| date | date | 交易日 |
| close | float | 指数收盘价 |

可选字段：

| 字段 | 类型 | 说明 |
|---|---:|---|
| symbol | str | 指数代码 |

## 4. membership.csv，可选

如果需要严格使用历史指数成分股，需要提供成分股数据。

| 字段 | 类型 | 说明 |
|---|---:|---|
| date | date | 生效日期 |
| symbol | str | 股票代码 |
| in_universe | int/bool | 是否在股票池中 |

## 5. 避免未来函数

真实项目中，财务数据不能使用财报期末日期直接对齐调仓日，应使用公告日期或可获得日期。本项目使用 `merge_asof` 逻辑，只会选取调仓日之前最新可用的基本面数据。
