"""
留存因子终局绩效报告：用 vnpy.alpha 计算留存因子，输出指标表格与关键图表。

对 runs/final/factors.json 中每个因子：
1. 分段（train/valid/test，区间来自 config.json）IC / RankIC / IR / 方向胜率
2. 十分组单调性与多空（Q10-Q1）年化收益、波动、Sharpe、最大回撤、多头组换手率
3. 分年 RankIC（检验信号衰减）
4. 四张图：分组累计收益 / 累积 RankIC / 多空累计收益 / 分年 RankIC
报告开头另附三因子的 test 段多空净值对比图。

数据流（本脚本的核心结构）：因子面板经 analyze_factor 一次分组，产出两张
中间表——日度截面 IC 序列、十分组日收益表——之后所有指标与图表都从这两张
表派生（分段只需按日期切片），不做重复计算。

用法：
    python final_report.py
输出：
    runs/final/performance_report.md（表格 + 图集）
    runs/final/perf_detail.json（结构化明细）
    runs/final/charts/*.png
"""

from __future__ import annotations

import copy
import json
import re
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# matplotlib 默认字体（DejaVu Sans）不含中文字形，图例里的中文别名会变方框；
# 按常见桌面平台顺序指定中文字体（找不到的自动跳过），并修复负号显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

from eval_factors import load_base_dataset, load_config, load_filters

from vnpy.alpha.dataset import process_cs_norm, process_drop_na

BASE_DIR = Path(__file__).resolve().parent
CHART_DIR = BASE_DIR / "runs/final/charts"
TRADING_DAYS = 244                # A股年均交易日数，用于年化
COST_RATE = 0.001                 # 假设的单边交易成本（0.1%），用于估算换手拖累
SEG_COLORS = {"train": "#f0f0f0", "valid": "#d5e8d5", "test": "#d8e4f0"}


def compute_factors(factor_map: dict[str, str], config: dict) -> pd.DataFrame:
    """用 vnpy.alpha 管线一次算完全部留存因子 + label，返回长表 (datetime, vt_symbol, 因子..., label)。"""
    dataset = copy.deepcopy(load_base_dataset(config))
    for name, expr in factor_map.items():
        dataset.add_feature(name=name, expression=expr)
    dataset.set_label(config["label"])
    dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
    dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))
    dataset.prepare_data(load_filters(config), max_workers=config["workers"])

    cols = ["datetime", "vt_symbol"] + list(factor_map) + ["label"]
    result = dataset.result_df.select(cols).to_pandas()
    start = pd.Timestamp(config["start"])
    end = pd.Timestamp(config["end"])
    return result.loc[(result["datetime"] >= start) & (result["datetime"] <= end)].copy()


def analyze_factor(df: pd.DataFrame, name: str, q: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """单因子核心分析：只做一次分组，产出后续全部指标/图表依赖的中间对象。

    返回 (ic_daily, grp, top_members)：
    - ic_daily:    表①，日度截面 IC 表（ic / rank_ic 两列）
    - grp:         表②，十分组日收益表（行=datetime，列=Q1..Q10）
    - top_members: 多头组（Q10）每日成分集合，换手率用
    """
    panel = df.dropna(subset=[name, "label"]).copy()
    panel["group"] = panel.groupby("datetime")[name].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False) + 1
    )

    # 表①：日度截面 IC（Pearson + Spearman 两条）
    ic_daily = panel.groupby("datetime").apply(
        lambda g: pd.Series({
            "ic": g[name].corr(g["label"], method="pearson"),
            "rank_ic": g[name].corr(g["label"], method="spearman"),
        })
    ).astype(float)

    # 表②：各分组日平均前瞻收益
    grp = panel.groupby(["datetime", "group"])["label"].mean().unstack()
    grp.columns = [f"Q{c}" for c in grp.columns]

    # 多头组合每日成分（换手率用）
    top_members = panel[panel["group"] == q].groupby("datetime")["vt_symbol"].apply(set).sort_index()

    return ic_daily, grp, top_members


def ic_stats(ic_daily: pd.DataFrame) -> dict:
    """表① -> IC/RankIC 均值、IR、方向胜率（分段时传入按日期切片的子表）。"""
    ic = ic_daily["ic"].dropna()
    ric = ic_daily["rank_ic"].dropna()
    return {
        "days": int(len(ic_daily)),
        "ic_mean": float(ic.mean()),
        "ir": float(ic.mean() / ic.std()) if ic.std() else 0.0,
        "rank_ic_mean": float(ric.mean()),
        "rank_ir": float(ric.mean() / ric.std()) if ric.std() else 0.0,
        "directional_win_rate": float(((ric > 0) == (ric.mean() > 0)).sum() / len(ric)),
    }


def ls_stats(grp: pd.DataFrame) -> dict:
    """表② -> 分组单调性与多空（Q10-Q1）年化收益、波动、Sharpe、最大回撤。"""
    ls = (grp.iloc[:, -1] - grp.iloc[:, 0]).dropna()
    ann_ret = float(ls.mean() * TRADING_DAYS)
    ann_vol = float(ls.std() * np.sqrt(TRADING_DAYS))
    cum = (1 + ls).cumprod()
    q_means = grp.mean()

    return {
        "quantile_mean_label": {k: float(v) for k, v in q_means.items()},
        "monotonicity": float(pd.Series(q_means.to_numpy()).corr(
            pd.Series(range(1, len(q_means) + 1)), method="spearman")),
        "ls_annual_return": ann_ret,
        "ls_annual_vol": ann_vol,
        "ls_sharpe": ann_ret / ann_vol if ann_vol else 0.0,
        "ls_max_drawdown": float((cum / cum.cummax() - 1).min()),
        "long_annual_return": float(grp.iloc[:, -1].mean() * TRADING_DAYS),
    }


def top_turnover(members: pd.Series) -> float:
    """多头组日均换手：相邻两日成分集合的 Jaccard 距离均值。"""
    if len(members) < 2:
        return 0.0
    overlap = [len(a & b) / len(a | b) for a, b in zip(members.iloc[1:], members.iloc[:-1])]
    return float(1 - np.mean(overlap))


def yearly_rank_ic(ic_daily: pd.DataFrame) -> dict[int, float]:
    """表① -> 分年 RankIC（检验信号是否随时间衰减）。"""
    ric = ic_daily["rank_ic"].dropna()
    return {int(y): float(v) for y, v in ric.groupby(ric.index.year).mean().items()}


def fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def add_segment_shading(ax: plt.Axes, segments: dict[str, tuple[str, str]]) -> None:
    """按 train/valid/test 分段加底色（区间来自 config.json），直观展示样本外位置。"""
    for seg_name, (start, end) in segments.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=SEG_COLORS.get(seg_name, "#eeeeee"), zorder=0)
        ax.text(pd.Timestamp(start), ax.get_ylim()[1], seg_name, fontsize=8, va="top")


def make_charts(name: str, ic_daily: pd.DataFrame, grp: pd.DataFrame,
                segments: dict[str, tuple[str, str]], out_dir: Path) -> list[Path]:
    """单因子四张图：分组累计收益、累积 RankIC、多空累计收益、分年 RankIC。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"\W+", "_", name)
    paths: list[Path] = []

    # 图1：十分组累计收益（Q1/Q10 加粗，一眼检验单调性）
    # 用 cumsum 而非 cumprod：这里只关心次日开盘缺口的组间 spread，
    # 线性累积更直观（口径同 alphalens 的 quantile returns）。
    cum_grp = grp.cumsum()
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("RdYlGn")
    for i, q in enumerate(grp.columns):
        lw = 2.2 if q in ("Q1", "Q10") else 0.9
        ax.plot(cum_grp.index, cum_grp[q], lw=lw,
                color=cmap(i / (len(grp.columns) - 1)), label=q)
    ax.set_title(f"{safe}: cumulative next-open gap by decile (daily sum)")
    ax.set_ylabel("cum return")
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out_dir / f"{safe}_decile_cum.png"))

    # 图2：累积 RankIC（斜率=该时期日均 RankIC，看信号衰减）
    cum_ic = ic_daily["rank_ic"].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cum_ic.index, cum_ic.values, lw=1.2, color="#1f77b4")
    add_segment_shading(ax, segments)
    ax.set_title(f"{safe}: cumulative RankIC")
    ax.set_ylabel("cum RankIC")
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out_dir / f"{safe}_cum_ic.png"))

    # 图3：多空（Q10-Q1）累计收益（cumsum 口径，同图1）
    ls = (grp.iloc[:, -1] - grp.iloc[:, 0]).dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ls.index, ls.cumsum().values, lw=1.2, color="#1f77b4")
    add_segment_shading(ax, segments)
    ax.set_title(f"{safe}: Q10-Q1 cumulative next-open gap (daily sum)")
    ax.set_ylabel("cum return")
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out_dir / f"{safe}_ls_cum.png"))

    # 图4：分年 RankIC
    yr = yearly_rank_ic(ic_daily)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    vals = list(yr.values())
    ax.bar([str(y) for y in yr], [v * 100 for v in vals],
           color=["#27ae60" if v > 0 else "#c0392b" for v in vals])
    ax.set_title(f"{safe}: yearly RankIC (%)")
    ax.grid(alpha=0.3, axis="y")
    paths.append(_save(fig, out_dir / f"{safe}_yearly_ic.png"))

    return paths


def make_comparison_chart(ls_map: dict[str, pd.Series], segments: dict[str, tuple[str, str]],
                          out_dir: Path) -> Path:
    """三因子 test 段多空（Q10-Q1）累计收益对比——报告总览的主图。"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, ls in ls_map.items():
        ax.plot(ls.index, ls.cumsum().values, lw=1.4, label=label)
    start, end = segments["test"]
    ax.set_title(f"Q10-Q1 cumulative next-open gap — test segment ({start[:4]}-{end[:4]}), daily sum")
    ax.set_ylabel("cum return")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir / "comparison_ls_cum.png")


def build_section(alias: str, key: str, expr: str, seg_ic: dict, perf: dict,
                  seg_perf: dict, yr: dict[int, float], cost_drag: float) -> list[str]:
    """单个因子的 markdown 小节：分段 IC 表 + 全样本多空表 + 分段多空表 + 分年 RankIC。"""
    lines = [f"## {alias}（{key}）", "", f"表达式：{expr}", ""]

    lines += ["### 分段 IC 指标", "",
              "| 段 | 天数 | IC | IR | RankIC | RankIR | 方向胜率 |",
              "|---|---|---|---|---|---|---|"]
    for seg_name, m in seg_ic.items():
        lines.append(
            f"| {seg_name} | {m['days']} | {m['ic_mean']:+.4f} | {m['ir']:+.3f} "
            f"| {m['rank_ic_mean']:+.4f} | {m['rank_ir']:+.3f} | {m['directional_win_rate']:.2%} |"
        )

    q_str = " / ".join(fmt_pct(v) for v in perf["quantile_mean_label"].values())
    lines += ["", "### 全样本分组与多空表现", "",
              "| 指标 | 数值 |", "|---|---|",
              f"| 分组平均日收益 Q1→Q10 | {q_str} |",
              f"| 单调性（Q 序号 Spearman） | {perf['monotonicity']:+.2f} |",
              f"| 多空年化收益 | {fmt_pct(perf['ls_annual_return'])} |",
              f"| 多空年化波动 | {fmt_pct(perf['ls_annual_vol'])} |",
              f"| 多空 Sharpe | {perf['ls_sharpe']:.2f} |",
              f"| 多空最大回撤 | {fmt_pct(perf['ls_max_drawdown'])} |",
              f"| 多头(Q10)年化 | {fmt_pct(perf['long_annual_return'])} |",
              f"| 多头组日均换手率 | {perf['top_quantile_daily_turnover']:.1%} |",
              "",
              f"> 成本提示：多头组日均换手 {perf['top_quantile_daily_turnover']:.1%}，按单边 "
              f"{COST_RATE:.1%}、双边每日调仓估算，年化拖累约 {fmt_pct(cost_drag)}，扣除后多空"
              f"年化约 {fmt_pct(perf['ls_annual_return'] - cost_drag)}。IC 类指标 ≠ 扣费后组合收益。",
              ""]

    lines += ["### 分段多空表现", "",
              "| 段 | 多空年化 | 多空波动 | 多空 Sharpe | 最大回撤 | 多头(Q10)年化 | 多头组换手 |",
              "|---|---|---|---|---|---|---|"]
    for seg_name, sp in seg_perf.items():
        lines.append(
            f"| {seg_name} | {fmt_pct(sp['ls_annual_return'])} | {fmt_pct(sp['ls_annual_vol'])} "
            f"| {sp['ls_sharpe']:.2f} | {fmt_pct(sp['ls_max_drawdown'])} "
            f"| {fmt_pct(sp['long_annual_return'])} | {sp['top_quantile_daily_turnover']:.1%} |"
        )

    lines += ["", "### 分年 RankIC", "", "| 年份 | RankIC |", "|---|---|"]
    for y, v in yr.items():
        lines.append(f"| {y} | {v:+.4f} |")
    pos_years = sum(1 for v in yr.values() if v > 0)
    lines += ["", f"**{pos_years}/{len(yr)} 年 RankIC 为正**", ""]
    return lines


def main() -> None:
    config = load_config()
    raw: dict = json.loads((BASE_DIR / "runs/final/factors.json").read_text(encoding="utf-8"))
    segments = {
        "train": tuple(config["train_period"]),
        "valid": tuple(config["valid_period"]),
        "test": tuple(config["test_period"]),
    }

    print(f"用 vnpy.alpha 计算 {len(raw)} 个留存因子 ...")
    df = compute_factors({k: v["expression"] for k, v in raw.items()}, config)

    detail: dict = {}
    sections: list[str] = []      # 每个因子的小节
    summary_rows: list[str] = []  # 顶部总览表
    test_ls_map: dict[str, pd.Series] = {}

    for key, item in raw.items():
        expr, alias = item["expression"], item.get("name", key)
        print(f"分析 {alias} ...")

        ic_daily, grp, members = analyze_factor(df, key)

        # 全样本多空 + 换手率 + 成本拖累估算
        perf = ls_stats(grp)
        turnover = top_turnover(members)
        perf["top_quantile_daily_turnover"] = turnover
        cost_drag = turnover * 2 * COST_RATE * TRADING_DAYS

        # 分段指标：两张中间表按日期切片即可，不重新计算
        seg_ic, seg_perf = {}, {}
        for seg_name, (start, end) in segments.items():
            seg_ic[seg_name] = ic_stats(ic_daily.loc[start:end])
            seg_perf[seg_name] = ls_stats(grp.loc[start:end])
            seg_perf[seg_name]["top_quantile_daily_turnover"] = top_turnover(members.loc[start:end])

        yr = yearly_rank_ic(ic_daily)

        # test 段多空序列收入对比图（图例带 test RankIR）
        test_ls_map[f"{alias} (test RankIR {item.get('test_rank_ir', 0):.2f})"] = \
            (grp.iloc[:, -1] - grp.iloc[:, 0]).loc[segments["test"][0]:segments["test"][1]]

        detail[key] = {
            "name": alias,
            "expression": expr,
            "segments": seg_ic,
            "performance": perf,
            "segment_performance": seg_perf,
            "yearly_rank_ic": yr,
        }

        sections += build_section(alias, key, expr, seg_ic, perf, seg_perf, yr, cost_drag)
        sp = seg_perf["test"]
        summary_rows.append(
            f"| {alias} | {item.get('train_rank_ir', 0):+.2f} | {item.get('valid_rank_ir', 0):+.2f} "
            f"| {item.get('test_rank_ir', 0):+.2f} | {sp['ls_sharpe']:.2f} |"
        )

        for p in make_charts(key, ic_daily, grp, segments, CHART_DIR):
            sections += [f"![{p.stem}](charts/{p.name})", ""]

    cmp_path = make_comparison_chart(test_ls_map, segments, CHART_DIR)

    lines: list[str] = [
        "# 留存因子终局绩效分析报告",
        "",
        "- 因子值与前瞻 label 均由 vnpy.alpha（AlphaDataset 表达式引擎）计算",
        "- 股票池：沪深300 历史成分股日线；label = 次日开盘价 / 当日收盘价 - 1",
        "- 分组：每日按因子值 10 分组（qcut）；多空 = Q10 均值 − Q1 均值；年化按 244 交易日",
        "",
        "## 因子总览",
        "",
        "| 因子 | train RankIR | valid RankIR | test RankIR | test 多空 Sharpe |",
        "|---|---|---|---|---|",
        *summary_rows,
        "",
        "test 段为三个因子的共同样本外确认区间，多空净值对比如下：",
        "",
        f"![三因子 test 段多空对比](charts/{cmp_path.name})",
        "",
        *sections,
    ]

    out_md = BASE_DIR / "runs" / "final" / "performance_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    (BASE_DIR / "runs" / "final" / "perf_detail.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"报告已写入: {out_md}")


if __name__ == "__main__":
    main()
