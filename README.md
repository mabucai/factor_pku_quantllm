# 日线涨停事件与次日高开：自动化因子挖掘记录

本项目是北京大学 VeighNa 量化平台课程的因子研究作业，使用 `vnpy.alpha` 检验：

> 当股票当天快速上涨并收盘封板时，次日是否更容易相对高开？

当前数据只有日线，因此研究的是“可成交涨停日线代理”，而不是分钟级封板速度。预测标签为：

```text
次日开盘价 / 当日收盘价 - 1
```

## 研究设计

- 股票池：沪深300历史成分股日线。
- train：2008–2014。
- valid：2015–2016。
- test：2017–2020-08。
- 第一轮：10个候选，拆解封板事件、日内强度、量能、事前趋势和首板结构。
- 第二轮：12个候选，检查涨幅阈值、收盘封板阈值、可成交振幅和首板窗口参数平台。
- 指标：日度截面 IC、RankIC、IR、RankIR；终局增加十分组、Q10-Q1、换手和分年分析。

完整研究反思见 [notes.md](notes.md)，终局报告见 [performance_report.md](runs/final/performance_report.md)。

## 最终因子

| 因子 | Train RankIR | Valid RankIR | Test RankIR | Test Q10-Q1 Sharpe |
|---|---:|---:|---:|---:|
| 严格收盘封板代理 | 0.785 | 0.772 | 0.745 | 5.42 |
| 近5日首板代理 | 0.754 | 0.744 | 0.722 | 4.23 |
| 封板 × 日内上涨强度 | 0.714 | 0.741 | 0.716 | 5.32 |

主推因子 `strict_close_board` 要求：当日涨幅超过9.5%，收盘价非常接近日内最高价，同时日内最低价低于收盘价，以排除近似一字板。

三个留存因子在2008–2023的16个年度中年度 RankIC 均为正。不过，这只能证明日线事件与次日开盘缺口存在稳定的截面关系，不能证明涨停价排队订单能够实际成交。

## 参数平台摘要

- 涨幅阈值9.0%–9.8%表现平滑。
- 收盘价越接近日内最高价，次日高开信号越强。
- 最低价过滤0.1%–1.0%差异很小。
- 首板回看3/5/10日的 valid RankIR 差异小于0.02，最终采用居中的5日。
- 成交量、成交额和事前5/20日趋势没有显著超过基础封板事件。

## 目录结构

```text
config.json                  数据区间、标签和数据路径
candidates.json              第一轮10个候选
metrics.json                 第一轮评估结果
notes.md                     研究反思、参数平台与选择理由
round_02/
  candidates.json            第二轮12个候选
  metrics.json               第二轮评估结果
eval_factors.py              批量因子评估
final_report.py              终局指标和图表生成
runs/final/
  factors.json               最终3因子
  perf_detail.json           结构化绩效明细
  performance_report.md      深度报告
  charts/                    13张报告图表
```

## 复现

安装依赖：

```bash
pip install -r requirements.txt --index-url https://pypi.vnpy.com
```

按 [data/README.md](data/README.md) 放置课程数据后，在仓库根目录运行：

```bash
# 第一轮
python eval_factors.py candidates.json

# 第二轮
python eval_factors.py round_02/candidates.json

# 最终报告
python final_report.py
```

运行后会分别更新第一轮/第二轮 `metrics.json`，并在 `runs/final/` 生成结构化明细、Markdown报告和图表。

## 局限性

- 日线无法观察首次封板时间、炸板次数、封单量和排队成交。
- `low / close` 只能排除近似一字板，不能证明真实可成交性。
- 涨停制度、ST状态和不同板块的10%/20%涨跌停规则没有完整建模。
- 因子是稀疏事件变量，十分组分析应理解为顶部事件组相对无事件组，而不是连续因子的完整单调性。
- test 已在研究过程中被查看，后续继续调参会形成测试集污染。
