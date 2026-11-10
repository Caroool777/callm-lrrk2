#!/usr/bin/env python
"""Create a CALLM benchmark performance landscape figure.

The script prefers a curated summary CSV, but can also discover metrics.json
files under MultiResult and assemble a best-effort benchmark table.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - exercised only when seaborn is absent
    sns = None


TASK_FAMILY_ORDER = [
    "Property",
    "Similarity",
    "Rediscovery",
    "Isomer",
    "MPO",
    "Structure",
    "LRRK2",
    "Other",
]

TASK_FAMILY_MAP = {
    "Property": {"qed", "jnk3", "drd2", "gsk3b"},
    "Similarity": {"albuterol_similarity", "mestranol_similarity"},
    "Rediscovery": {
        "celecoxib_rediscovery",
        "thiothixene_rediscovery",
        "troglitazone_rediscovery",
    },
    "Isomer": {"isomers_c7h8n2o2", "isomers_c9h10n2o2pf2cl"},
    "MPO": {
        "sitagliptin_mpo",
        "zaleplon_mpo",
        "amlodipine_mpo",
        "fexofenadine_mpo",
        "osimertinib_mpo",
        "perindopril_mpo",
        "ranolazine_mpo",
    },
    "LRRK2": {"lrrk2_wt_activity_mpo", "lrrk2_wt_lead_repair_mpo"},
    "Structure": {"median1", "median2", "deco_hop", "scaffold_hop", "valsartan_smarts"},
}

KEY_FIELDS = [
    "method",
    "task",
    "task_family",
    "seed",
    "auc_top_1",
    "auc_top_10",
    "auc_top_100",
    "top_1_score",
    "top_10_avg_score",
    "top_100_avg_score",
    "hypervolume",
    "spacing",
    "spread",
    "pareto_diversity",
    "total_oracle_calls",
    "total_evaluated_unique_molecules",
]

NUMERIC_FIELDS = [
    "seed",
    "auc_top_1",
    "auc_top_10",
    "auc_top_100",
    "top_1_score",
    "top_10_avg_score",
    "top_10_score",
    "top_100_avg_score",
    "top_100_score",
    "hypervolume",
    "spacing",
    "spread",
    "pareto_diversity",
    "diversity",
    "total_oracle_calls",
    "total_evaluated_unique_molecules",
    "n_objectives",
]


def canonical_name(value: object) -> str:
    """Normalize task/method names while keeping them readable."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    text = text.replace("\xa0", " ")
    text = text.replace("β", "b").replace("Β", "b")
    text = re.sub(r"\s+", "_", text)
    text = text.replace("-", "_")
    return text.lower()


def clean_label(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def clean_task_from_stem(stem: str) -> str:
    task = canonical_name(stem)
    for suffix in (
        "_pareto_metrics",
        "_metrics",
        "_top_molecules",
        "_results",
        "_result",
        "_seed",
    ):
        if task.endswith(suffix):
            task = task[: -len(suffix)]
    task = re.sub(r"_results_seed\d+$", "", task)
    task = re.sub(r"_seed\d+$", "", task)
    return task


def infer_task_family(task: object) -> str:
    """Infer the benchmark family from a task name."""
    task_name = canonical_name(task)
    for family, tasks in TASK_FAMILY_MAP.items():
        if task_name in tasks:
            return family
    return "Other"


def infer_task_family_from_type(task: object, task_type: object) -> str:
    """Infer family from task name, using broad table task type as a fallback."""
    family = infer_task_family(task)
    broad_type = canonical_name(task_type)
    if family != "Other":
        return family
    if broad_type.startswith("property"):
        return "Property"
    if broad_type.startswith("structure"):
        return "Structure"
    return "Other"


def parse_score_cell(value: object) -> tuple[float, float]:
    """Parse benchmark cells such as '0.941 ± 0.000' into mean and std."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan, np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), np.nan

    text = clean_label(value)
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
    if not numbers:
        return np.nan, np.nan
    mean = float(numbers[0])
    std = float(numbers[1]) if len(numbers) > 1 else np.nan
    return mean, std


def is_benchmark_wide_dataframe(df: pd.DataFrame) -> bool:
    columns = [clean_label(col).lower() for col in df.columns]
    has_task = any("method objective" in col or "objective" in col for col in columns)
    has_method_col = any(col == "method" for col in columns)
    return has_task and not has_method_col


def benchmark_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the PMO-style wide benchmark table to a long metrics table."""
    df = df.copy()
    df.columns = [clean_label(col) for col in df.columns]

    task_type_col = next(
        (col for col in df.columns if "task type" in col.lower()), df.columns[0]
    )
    task_col = next(
        (col for col in df.columns if "method objective" in col.lower()),
        df.columns[1],
    )
    method_cols = [col for col in df.columns if col not in {task_type_col, task_col}]

    df[task_type_col] = df[task_type_col].ffill()
    records = []
    for _, row in df.iterrows():
        task = canonical_name(row.get(task_col))
        if not task or task.startswith("total") or task.startswith("rank"):
            continue
        task_type = row.get(task_type_col)
        task_family = infer_task_family_from_type(task, task_type)
        for method in method_cols:
            mean, std = parse_score_cell(row.get(method))
            if np.isnan(mean):
                continue
            records.append(
                {
                    "method": clean_label(method),
                    "task": task,
                    "task_family": task_family,
                    "seed": np.nan,
                    "auc_top_10": mean,
                    "auc_top_10_std": std,
                    "top_10_avg_score": mean,
                    "top_10_avg_score_std": std,
                }
            )
    return pd.DataFrame.from_records(records)


def read_summary_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        xl = pd.ExcelFile(path)
        for sheet_name in xl.sheet_names:
            sheet = pd.read_excel(path, sheet_name=sheet_name)
            if not sheet.empty:
                print(f"[load] Reading Excel sheet: {path}::{sheet_name}")
                return sheet
        return pd.DataFrame()
    print(f"[load] Reading summary CSV: {path}")
    return pd.read_csv(path)


def infer_seed_from_path(path: Path) -> float:
    for part in reversed(path.parts):
        match = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:$|[_-])", part.lower())
        if match:
            return float(match.group(1))
    return np.nan


def infer_task_from_path(path: Path, payload: dict) -> str:
    for key in ("task", "task_name", "benchmark", "objective", "oracle_name"):
        task = payload.get(key)
        if task:
            return canonical_name(task)

    stem_task = clean_task_from_stem(path.stem)
    if stem_task and stem_task not in {"metrics", "pareto_metrics"}:
        return stem_task

    for part in reversed(path.parent.parts):
        task = clean_task_from_stem(part)
        if task not in {"singleresult", "multiresult", "resultbelief", "oracle"}:
            return task
    return ""


def looks_task_like(name: str, task: str) -> bool:
    if not name:
        return False
    if task and (name == task or task in name or name in task):
        return True
    return bool(
        name.endswith("_top_molecules")
        or name.endswith("_results")
        or re.search(r"_results_seed\d+$", name)
        or name.endswith("_metrics")
    )


def infer_method_from_path(path: Path, task: str) -> str:
    """Infer a method from the path, falling back to CALLM for task-only layouts."""
    skip = {
        "multiresult",
        "singleresult",
        "resultbelief",
        "oracle",
        "figures",
        "logs",
    }
    known_method_tokens = (
        "callm",
        "baseline",
        "ablation",
        "random",
        "graphga",
        "molga",
        "mso",
        "reinvent",
        "mars",
        "moler",
        "dst",
        "gpbo",
        "bo",
        "ga",
        "mcts",
    )

    for part in reversed(path.parent.parts):
        name = canonical_name(part)
        if not name or name in skip:
            continue
        if re.match(r"run[_-]?seed\d+$", name):
            continue
        if name.endswith("_runs") or name.endswith("_logs"):
            continue
        if looks_task_like(name, task):
            continue
        if any(token in name for token in known_method_tokens):
            return name

    for part in reversed(path.parent.parts):
        name = canonical_name(part)
        if not name or name in skip:
            continue
        if re.match(r"run[_-]?seed\d+$", name):
            continue
        if name.endswith("_runs") or name.endswith("_logs"):
            continue
        if looks_task_like(name, task):
            continue
        return name

    return "CALLM"


def extract_record_from_json(path: Path, result_root: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        warnings.warn(f"Skipping unreadable JSON {path}: {exc}")
        return None

    if not isinstance(payload, dict):
        warnings.warn(f"Skipping non-object JSON {path}")
        return None

    task = infer_task_from_path(path, payload)
    method = payload.get("method") or payload.get("algorithm") or payload.get("model")
    if not method:
        method = infer_method_from_path(path, task)

    record = {field: payload.get(field, np.nan) for field in KEY_FIELDS}
    record["method"] = str(method).strip() if str(method).strip() else "CALLM"
    record["task"] = task
    record["task_family"] = payload.get("task_family") or infer_task_family(task)
    record["seed"] = payload.get("seed", payload.get("random_seed", infer_seed_from_path(path)))
    record["source_file"] = str(path.relative_to(result_root))
    record["top_10_score"] = payload.get("top_10_score", np.nan)
    record["top_100_score"] = payload.get("top_100_score", np.nan)
    record["n_objectives"] = payload.get("n_objectives", np.nan)
    record["pareto_diversity"] = payload.get(
        "pareto_diversity", payload.get("diversity", np.nan)
    )

    useful_metrics = [
        "auc_top_10",
        "top_10_avg_score",
        "top_10_score",
        "hypervolume",
        "top_1_score",
    ]
    if not task or not any(pd.notna(record.get(metric)) for metric in useful_metrics):
        warnings.warn(f"Skipping {path}: no task or no usable performance metric")
        return None
    return record


def load_metrics(result_root: str | Path, summary_csv: str | Path) -> pd.DataFrame:
    """Load metrics from summary CSV or recursively discovered metrics JSON files."""
    result_root = Path(result_root)
    summary_csv = Path(summary_csv)

    if summary_csv.exists():
        df = read_summary_table(summary_csv)
        if is_benchmark_wide_dataframe(df):
            df = benchmark_wide_to_long(df)
            print(f"[load] Parsed wide benchmark table into {len(df)} long rows")
        for field in KEY_FIELDS:
            if field not in df.columns:
                df[field] = np.nan
        inferred_family = df["task"].map(infer_task_family)
        existing_family = df["task_family"].astype("string")
        missing_family = existing_family.isna() | existing_family.str.strip().eq("")
        df.loc[missing_family, "task_family"] = inferred_family[missing_family]
        return df

    print(f"[load] Summary CSV not found; scanning metrics JSON under {result_root}")
    json_paths = sorted(result_root.rglob("*metrics.json"))
    records = []
    for path in json_paths:
        record = extract_record_from_json(path, result_root)
        if record is not None:
            records.append(record)

    if not records:
        warnings.warn(f"No usable metrics JSON files found under {result_root}")
        return pd.DataFrame(columns=KEY_FIELDS)

    df = pd.DataFrame.from_records(records)
    for field in KEY_FIELDS:
        if field not in df.columns:
            df[field] = np.nan
    return df


def choose_final_metric_values(df: pd.DataFrame, final_metric: str) -> pd.Series:
    if final_metric != "auto":
        if final_metric not in df.columns:
            warnings.warn(f"Requested final metric '{final_metric}' is absent; using NaN")
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[final_metric], errors="coerce")

    hypervolume = pd.to_numeric(
        df["hypervolume"] if "hypervolume" in df.columns else pd.Series(np.nan, index=df.index),
        errors="coerce",
    )
    n_objectives = pd.to_numeric(
        df["n_objectives"] if "n_objectives" in df.columns else pd.Series(np.nan, index=df.index),
        errors="coerce",
    )
    task_family = df.get("task_family", pd.Series("", index=df.index)).astype(str)
    is_multiobjective = hypervolume.notna() & (
        (n_objectives.fillna(0) > 1) | task_family.isin(["MPO", "LRRK2"])
    )
    final_values = pd.to_numeric(df.get("top_10_avg_score", np.nan), errors="coerce")

    if "top_10_score" in df.columns:
        final_values = final_values.fillna(
            pd.to_numeric(df["top_10_score"], errors="coerce")
        )
    final_values = final_values.where(~is_multiobjective, hypervolume)
    return final_values


def normalize_within_task(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    """Min-max normalize a metric within each task, preserving missing values."""
    df = df.copy()
    df[out_col] = np.nan

    for task, idx in df.groupby("task", dropna=False).groups.items():
        values = pd.to_numeric(df.loc[idx, value_col], errors="coerce")
        finite = values.dropna()
        if finite.empty:
            continue
        vmin = finite.min()
        vmax = finite.max()
        if np.isclose(vmax, vmin):
            df.loc[finite.index, out_col] = 0.5
        else:
            df.loc[finite.index, out_col] = (finite - vmin) / (vmax - vmin)
    return df


def aggregate_metrics(
    df: pd.DataFrame,
    primary_metric: str = "auc_top_10",
    final_metric: str = "auto",
) -> pd.DataFrame:
    """Aggregate multiple seeds for each method-task pair and normalize by task."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "task",
                "task_family",
                "mean_auc_top_10",
                "relative_auc_top_10",
                "mean_final_metric",
                "relative_final_metric",
            ]
        )

    df = df.copy()
    for field in NUMERIC_FIELDS:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")
    if primary_metric not in df.columns:
        warnings.warn(f"Primary metric '{primary_metric}' is absent; using NaN")
        df[primary_metric] = np.nan

    df["task"] = df["task"].map(canonical_name)
    df["task_family"] = df["task_family"].fillna(df["task"].map(infer_task_family))
    df["task_family"] = df.apply(
        lambda row: row["task_family"] if pd.notna(row["task_family"]) else infer_task_family(row["task"]),
        axis=1,
    )
    df.loc[df["task_family"].eq("") | df["task_family"].isna(), "task_family"] = df[
        "task"
    ].map(infer_task_family)
    df["method"] = df["method"].fillna("CALLM").astype(str).str.strip()
    df.loc[df["method"].eq(""), "method"] = "CALLM"
    df["final_metric_value"] = choose_final_metric_values(df, final_metric)
    if final_metric == "auto":
        primary_values = pd.to_numeric(df[primary_metric], errors="coerce")
        missing_final = df["final_metric_value"].isna() & primary_values.notna()
        if missing_final.any():
            warnings.warn(
                "No separate final metric was available for some rows; using the "
                f"primary metric '{primary_metric}' for final-performance plotting."
            )
            df.loc[missing_final, "final_metric_value"] = primary_values[missing_final]

    group_cols = ["method", "task", "task_family"]
    grouped = df.groupby(group_cols, dropna=False)
    aggregated = grouped.agg(
        mean_auc_top_10=(primary_metric, "mean"),
        std_auc_top_10=(primary_metric, "std"),
        n_auc_top_10=(primary_metric, lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        mean_final_metric=("final_metric_value", "mean"),
        std_final_metric=("final_metric_value", "std"),
        n_final_metric=("final_metric_value", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        n_seeds=("seed", lambda s: int(pd.to_numeric(s, errors="coerce").dropna().nunique())),
    )
    aggregated = aggregated.reset_index()
    aggregated = normalize_within_task(
        aggregated, "mean_auc_top_10", "relative_auc_top_10"
    )
    aggregated = normalize_within_task(
        aggregated, "mean_final_metric", "relative_final_metric"
    )
    return aggregated


def method_sort_key(method: str) -> tuple[int, str]:
    name = canonical_name(method)
    if name == "callm" or name.startswith("callm_"):
        tier = 0
    elif "ablation" in name or name.startswith("abl"):
        tier = 2
    elif any(token in name for token in ("baseline", "random", "control")):
        tier = 3
    else:
        tier = 1
    return tier, name


def task_sort_key(task: str, task_family: str) -> tuple[int, str]:
    family_rank = TASK_FAMILY_ORDER.index(task_family) if task_family in TASK_FAMILY_ORDER else 99
    return family_rank, canonical_name(task)


def ordered_methods(df: pd.DataFrame) -> list[str]:
    return sorted(df["method"].dropna().unique().tolist(), key=method_sort_key)


def ordered_tasks(df: pd.DataFrame) -> list[str]:
    task_meta = (
        df[["task", "task_family"]]
        .drop_duplicates()
        .sort_values(
            by=["task_family", "task"],
            key=lambda col: col.map(
                {family: i for i, family in enumerate(TASK_FAMILY_ORDER)}
            )
            if col.name == "task_family"
            else col,
        )
    )
    return [
        row.task
        for row in sorted(
            task_meta.itertuples(index=False),
            key=lambda row: task_sort_key(row.task, row.task_family),
        )
    ]


def family_for_tasks(df: pd.DataFrame, tasks: Iterable[str]) -> list[str]:
    lookup = (
        df.dropna(subset=["task"])
        .drop_duplicates("task")
        .set_index("task")["task_family"]
        .to_dict()
    )
    return [lookup.get(task, infer_task_family(task)) for task in tasks]


def build_matrix(
    df: pd.DataFrame,
    value_col: str,
    methods: list[str],
    tasks: list[str],
) -> pd.DataFrame:
    matrix = df.pivot_table(
        index="method", columns="task", values=value_col, aggfunc="mean"
    )
    return matrix.reindex(index=methods, columns=tasks)


def method_colors(methods: list[str]) -> list[str]:
    palette = list(mpl.colormaps["tab10"].colors) + list(mpl.colormaps["Set2"].colors)
    colors = []
    for i, method in enumerate(methods):
        name = canonical_name(method)
        if name == "callm" or name.startswith("callm_"):
            colors.append("#b2182b")
        elif "ablation" in name or name.startswith("abl"):
            colors.append("#ef8a62")
        elif any(token in name for token in ("baseline", "random", "control")):
            colors.append("#8c8c8c")
        else:
            colors.append(mpl.colors.to_hex(palette[i % len(palette)]))
    return colors


def short_method_label(method: str) -> str:
    lookup = {
        "augmented_memory": "AugMem",
        "graph_ga": "GraphGA",
        "gp_bo": "GP BO",
        "molleo_(biot5)": "BioT5",
        "molleo_(gpt_4)": "GPT-4",
        "molleo_(molstm)": "MolSTM",
        "reinvent": "REINVENT",
        "callm": "CALLM",
    }
    return lookup.get(canonical_name(method), method)


def plot_top_bars(
    ax: mpl.axes.Axes,
    df: pd.DataFrame,
    families: list[str],
    title: str,
    methods: list[str],
    value_col: str = "relative_auc_top_10",
) -> None:
    subset = df[df["task_family"].isin(families)].copy()
    colors = method_colors(methods)

    means = []
    sems = []
    for method in methods:
        values = pd.to_numeric(
            subset.loc[subset["method"] == method, value_col], errors="coerce"
        ).dropna()
        if values.empty:
            means.append(np.nan)
            sems.append(0.0)
        else:
            means.append(values.mean())
            sems.append(values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0)

    x = np.arange(len(methods))
    ax.bar(
        x,
        np.nan_to_num(means, nan=0.0),
        yerr=sems,
        color=colors,
        edgecolor="#333333",
        linewidth=0.35,
        capsize=2.0,
        width=0.72,
        zorder=3,
    )
    for xpos, mean in zip(x, means):
        if np.isnan(mean):
            ax.text(xpos, 0.05, "NA", ha="center", va="bottom", fontsize=6, color="#777777")

    ax.set_title(title, fontsize=9, pad=5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean relative score", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_method_label(method) for method in methods], rotation=45, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7, length=2)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color("#4d4d4d")


def add_family_strip(
    ax: mpl.axes.Axes,
    tasks: list[str],
    task_families: list[str],
) -> None:
    if not tasks:
        return
    colors = {
        "Property": "#dbe9f6",
        "Similarity": "#e5f5e0",
        "Rediscovery": "#f7f4d9",
        "Isomer": "#f3e5f5",
        "MPO": "#fde0dd",
        "Structure": "#e7e1ef",
        "LRRK2": "#e0ecf4",
        "Other": "#eeeeee",
    }
    y0 = 1.015
    height = 0.055
    start = 0
    while start < len(tasks):
        family = task_families[start]
        end = start + 1
        while end < len(tasks) and task_families[end] == family:
            end += 1
        ax.add_patch(
            mpl.patches.Rectangle(
                (start, y0),
                end - start,
                height,
                transform=ax.get_xaxis_transform(),
                facecolor=colors.get(family, "#eeeeee"),
                edgecolor="white",
                linewidth=0.6,
                clip_on=False,
            )
        )
        ax.text(
            (start + end) / 2,
            y0 + height + 0.006,
            family,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
            clip_on=False,
        )
        start = end


def plot_heatmap(
    ax: mpl.axes.Axes,
    matrix: pd.DataFrame,
    title: str,
    cmap: mpl.colors.Colormap,
    show_xticklabels: bool = True,
    task_families: list[str] | None = None,
) -> None:
    ax.set_facecolor("#f2f2f2")
    if matrix.empty:
        ax.text(0.5, 0.5, "No usable data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    if sns is not None:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar=False,
            linewidths=0.35,
            linecolor="white",
            square=False,
            mask=matrix.isna(),
        )
    else:
        data = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
        ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title, fontsize=11, loc="left", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(
        matrix.columns.tolist() if show_xticklabels else [],
        rotation=60,
        ha="right",
        fontsize=7,
    )
    ax.set_yticklabels(matrix.index.tolist(), rotation=0, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if task_families is not None:
        add_family_strip(ax, matrix.columns.tolist(), task_families)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
        }
    )
    if sns is not None:
        sns.set_theme(style="white", context="paper", font="Arial")


def plot_landscape(df: pd.DataFrame, output_dir: Path) -> None:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ordered_methods(df) if not df.empty else []
    tasks = ordered_tasks(df) if not df.empty else []
    task_families = family_for_tasks(df, tasks) if tasks else []

    cmap = mpl.colormaps.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f2f2f2")

    fig = plt.figure(figsize=(16, 9), dpi=300, constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=5,
        width_ratios=[1, 1, 1, 1, 0.045],
        height_ratios=[1.0, 1.75, 1.75],
        hspace=0.78,
        wspace=0.65,
    )

    families_present = set(df["task_family"].dropna().astype(str)) if not df.empty else set()
    fourth_bar = (
        (["LRRK2"], "LRRK2", "relative_final_metric")
        if "LRRK2" in families_present
        else (["Isomer", "Structure", "Other"], "Structure / Other", "relative_auc_top_10")
    )
    bar_specs = [
        (["Property"], "Property", "relative_auc_top_10"),
        (["Similarity", "Rediscovery"], "Similarity + Rediscovery", "relative_auc_top_10"),
        (["MPO"], "MPO", "relative_auc_top_10"),
        fourth_bar,
    ]
    for col, (families, title, value_col) in enumerate(bar_specs):
        ax = fig.add_subplot(gs[0, col])
        plot_top_bars(ax, df, families, title, methods, value_col=value_col)

    ax_auc = fig.add_subplot(gs[1, 0:4])
    ax_final = fig.add_subplot(gs[2, 0:4])
    cax = fig.add_subplot(gs[1:3, 4])

    auc_matrix = build_matrix(df, "relative_auc_top_10", methods, tasks)
    final_matrix = build_matrix(df, "relative_final_metric", methods, tasks)
    plot_heatmap(
        ax_auc,
        auc_matrix,
        "Relative AUC Top-10",
        cmap,
        show_xticklabels=False,
        task_families=task_families,
    )
    plot_heatmap(
        ax_final,
        final_matrix,
        "Relative final performance",
        cmap,
        show_xticklabels=True,
        task_families=None,
    )

    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cax)
    colorbar.set_label("Relative score", fontsize=8)
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.outline.set_linewidth(0.5)

    fig.suptitle(
        "CALLM benchmark performance landscape",
        fontsize=13,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.subplots_adjust(left=0.075, right=0.94, top=0.9, bottom=0.18)

    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"figure_callm_performance_landscape.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[save] {path}")
    plt.close(fig)


def print_data_report(df: pd.DataFrame, processed: pd.DataFrame) -> None:
    if df.empty:
        print("[warning] No usable records were loaded.")
        return

    n_tasks = df["task"].dropna().nunique() if "task" in df else 0
    n_methods = df["method"].dropna().nunique() if "method" in df else 0
    n_seeds = pd.to_numeric(df.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().nunique()
    print(f"[data] loaded records: {len(df)}")
    print(f"[data] tasks: {n_tasks}, methods: {n_methods}, seeds: {int(n_seeds)}")

    missing = {}
    for field in KEY_FIELDS:
        if field not in df.columns:
            missing[field] = "column absent"
        else:
            count = int(df[field].isna().sum())
            if count:
                missing[field] = f"{count}/{len(df)} missing"
    if missing:
        print("[data] missing key fields:")
        for field, message in missing.items():
            print(f"  - {field}: {message}")
    else:
        print("[data] no missing key fields detected")

    if processed.empty:
        print("[warning] Aggregated table is empty; figure will contain placeholders.")
        return
    if processed["relative_auc_top_10"].notna().sum() == 0:
        print("[warning] No usable AUC Top-10 values were available.")
    if processed["relative_final_metric"].notna().sum() == 0:
        print("[warning] No usable final performance values were available.")
    method_counts = Counter(processed["method"])
    if len(method_counts) <= 1:
        print(
            "[warning] Only one method was found. Within-task normalization will be 0.5 "
            "for available values, so the figure is best interpreted as coverage rather "
            "than a method comparison."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a CALLM benchmark performance landscape figure."
    )
    parser.add_argument("--result-root", default="MultiResult")
    parser.add_argument(
        "--summary-csv", default="MultiResult/callm_summary_metrics.csv"
    )
    parser.add_argument("--output-dir", default="MultiResult/figures")
    parser.add_argument("--primary-metric", default="auc_top_10")
    parser.add_argument("--final-metric", default="auto")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    processed_csv = result_root / "callm_landscape_processed.csv"

    df = load_metrics(result_root, args.summary_csv)
    processed = aggregate_metrics(
        df,
        primary_metric=args.primary_metric,
        final_metric=args.final_metric,
    )
    print_data_report(df, processed)

    processed_columns = [
        "method",
        "task",
        "task_family",
        "mean_auc_top_10",
        "relative_auc_top_10",
        "mean_final_metric",
        "relative_final_metric",
        "std_auc_top_10",
        "std_final_metric",
        "n_auc_top_10",
        "n_final_metric",
        "n_seeds",
    ]
    for column in processed_columns:
        if column not in processed.columns:
            processed[column] = np.nan
    processed = processed[processed_columns]
    processed = processed.sort_values(
        by=["task_family", "task", "method"],
        key=lambda col: col.map({family: i for i, family in enumerate(TASK_FAMILY_ORDER)})
        if col.name == "task_family"
        else col,
    )
    processed.to_csv(processed_csv, index=False)
    print(f"[save] {processed_csv}")

    if len(processed) < 2:
        print("[warning] Very little data available; generated figure may be sparse.")
    plot_landscape(processed, output_dir)


if __name__ == "__main__":
    main()
