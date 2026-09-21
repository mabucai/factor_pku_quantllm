"""
因子绩效评估脚本：读取候选因子 JSON，基于 vnpy.alpha 计算分段绩效指标。

用法：
    python eval_factors.py runs/round_01/candidates.json

输入 JSON 格式（支持 ```json 围栏）：
    {"factor_1": "ts_corr(close, volume, 5)", "factor_2": "..."}

输出：
    - 同目录下 metrics.json（结构化结果，供 LLM 读取决策）
    - 控制台摘要表格
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from vnpy.trader.constant import Interval

from vnpy.alpha import AlphaDataset, AlphaLab
from vnpy.alpha.dataset import process_cs_norm, process_drop_na

BASE_DIR = Path(__file__).resolve().parent


def load_config() -> dict[str, Any]:
    path: Path = BASE_DIR / "config.json"
    config: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    lab_path = Path(config["lab_path"])
    if not lab_path.is_absolute():
        config["lab_path"] = str((BASE_DIR / lab_path).resolve())
    return config


def parse_factor_map(text: str) -> dict[str, str]:
    # 容错 LLM 输出常见的 ```json 围栏：整体剥掉首尾围栏标记
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    data: Any = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("候选文件顶层必须是 JSON 对象")

    factor_map: dict[str, str] = {}
    for name, expr in data.items():
        if isinstance(name, str) and isinstance(expr, str) and name.strip() and expr.strip():
            factor_map[name.strip()] = expr.strip()

    if not factor_map:
        raise ValueError("候选文件未包含有效因子表达式")
    return factor_map


def load_base_dataset(config: dict[str, Any]) -> AlphaDataset:
    """加载或构建基础数据集（构建后缓存为 pickle 到本项目 runs/cache）。"""
    cache_path: Path = BASE_DIR / config["cache_dir"] / "base_dataset.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"使用缓存数据集 {cache_path}（若修改过数据区间/成分股，请删除该文件后重建）")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # raw/ 目录只读：只从 lab 读数据，不调用 lab.save_dataset
    lab = AlphaLab(config["lab_path"])
    symbols = lab.load_component_symbols(
        index_symbol=config["index"],
        start=config["start"],
        end=config["end"],
    )
    df = lab.load_bar_df(
        vt_symbols=symbols,
        interval=Interval.DAILY,
        start=config["start"],
        end=config["end"],
        extended_days=config["extended_days"],
    )
    if df is None:
        raise RuntimeError("加载成分股数据失败")

    dataset = AlphaDataset(
        df=df,
        train_period=tuple(config["train_period"]),
        valid_period=tuple(config["valid_period"]),
        test_period=tuple(config["test_period"]),
    )

    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def load_filters(config: dict[str, Any]) -> dict | None:
    """加载成分过滤器（缓存到本项目，避免写 raw/）。"""
    cache_path = BASE_DIR / config["cache_dir"] / "filters.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    lab = AlphaLab(config["lab_path"])
    filters = lab.load_component_filters(
        index_symbol=config["index"],
        start=config["start"],
        end=config["end"],
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(filters, f)
    return filters


def evaluate(factor_map: dict[str, str], config: dict[str, Any]) -> list[dict[str, Any]]:
    """用 vnpy.alpha 管线计算因子，并按 train/valid（可选 test）分段计算 IC 指标。"""
    base = load_base_dataset(config)
    dataset: AlphaDataset = copy.deepcopy(base)

    for name, expr in factor_map.items():
        dataset.add_feature(name=name, expression=expr)

    dataset.set_label(config["label"])
    dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
    dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))
    dataset.prepare_data(load_filters(config), max_workers=config["workers"])

    if "label" not in dataset.result_df.columns:
        raise RuntimeError("标签列计算失败，请检查 label 表达式")

    segments: list[tuple[str, tuple[str, str]]] = [
        ("train", tuple(config["train_period"])),
        ("valid", tuple(config["valid_period"])),
    ]
    if config.get("eval_test"):
        segments.append(("test", tuple(config["test_period"])))

    results: list[dict[str, Any]] = []
    for name, expr in factor_map.items():
        item: dict[str, Any] = {"name": name, "expression": expr}

        try:
            if name not in dataset.result_df.columns:
                item["error"] = f"因子列 `{name}` 不存在（表达式解析或计算失败）"
                results.append(item)
                continue

            merged = dataset.result_df.select(["datetime", "vt_symbol", name, "label"])
            merged = merged.fill_nan(None).drop_nulls()
            if merged.height == 0:
                item["error"] = "无有效数据（全为 NaN）"
                results.append(item)
                continue

            pdf: pd.DataFrame = merged.to_pandas()

            for seg_name, (start, end) in segments:
                mask = (pdf["datetime"] >= pd.Timestamp(start)) & (pdf["datetime"] <= pd.Timestamp(end))
                seg = pdf.loc[mask]

                if seg.empty:
                    item.setdefault("segments", {})[seg_name] = {"error": "分段内无数据"}
                    continue

                daily = seg.groupby("datetime").apply(
                    lambda g: pd.Series(
                        {
                            "ic": g[name].corr(g["label"], method="pearson"),
                            "rank_ic": g[name].corr(g["label"], method="spearman"),
                        }
                    )
                ).astype(float)

                ic_mean = float(daily["ic"].mean())
                ic_std = float(daily["ic"].std())
                rank_mean = float(daily["rank_ic"].mean())
                rank_std = float(daily["rank_ic"].std())

                item.setdefault("segments", {})[seg_name] = {
                    "days": int(len(daily)),
                    "ic_mean": ic_mean,
                    "ic_std": ic_std,
                    "ir": ic_mean / ic_std if ic_std else 0.0,
                    "rank_ic_mean": rank_mean,
                    "rank_ic_std": rank_std,
                    "rank_ir": rank_mean / rank_std if rank_std else 0.0,
                    "ic_win_rate": float((daily["ic"].abs() > 0).sum() / len(daily)),
                    "directional_win_rate": float(
                        ((daily["ic"] > 0) == (ic_mean > 0)).sum() / len(daily)
                    ),
                }
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"

        results.append(item)

    return results


def format_summary(results: list[dict[str, Any]]) -> str:
    lines = ["因子绩效摘要（train / valid）", ""]
    for item in results:
        lines.append(f"[{item['name']}]  {item['expression']}")
        if "error" in item:
            lines.append(f"  错误: {item['error']}")
        else:
            for seg_name, seg in item.get("segments", {}).items():
                if "error" in seg:
                    lines.append(f"  {seg_name}: {seg['error']}")
                else:
                    lines.append(
                        f"  {seg_name}: IC={seg['ic_mean']:+.4f} IR={seg['ir']:+.3f} "
                        f"RankIC={seg['rank_ic_mean']:+.4f} RankIR={seg['rank_ir']:+.3f} "
                        f"方向胜率={seg['directional_win_rate']:.2%} 天数={seg['days']}"
                    )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    candidates_path = Path(sys.argv[1]).resolve()
    factor_map = parse_factor_map(candidates_path.read_text(encoding="utf-8"))
    config = load_config()

    print(f"评估 {len(factor_map)} 个因子: {sorted(factor_map)}")
    results = evaluate(factor_map, config)

    metrics_path = candidates_path.parent / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(format_summary(results))
    print(f"指标已写入: {metrics_path}")


if __name__ == "__main__":
    main()
