import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "auc_top_10",
    "auc_top_1",
    "auc_top_100",
    "top_1_score",
    "top_10_avg_score",
    "top_100_avg_score",
    "unique_scaffold_count",
    "scaffold_entropy",
    "mean_pairwise_tanimoto_distance",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize single-objective ablations")
    parser.add_argument("--result_root", default=os.path.join("MultiResult", "ablation_single"))
    return parser.parse_args()


def collect_metrics(result_root):
    rows = []
    for dirpath, _, filenames in os.walk(result_root):
        if "metrics.json" not in filenames:
            continue
        with open(os.path.join(dirpath, "metrics.json"), "r", encoding="utf-8") as f:
            row = json.load(f)
        row["output_dir"] = dirpath
        rows.append(row)
    return pd.DataFrame(rows)


def mean_std(df):
    grouped = df.groupby(["task_name", "ablation_mode"], dropna=False)
    rows = []
    for (task, mode), group in grouped:
        row = {"task_name": task, "ablation_mode": mode}
        for col in METRIC_COLUMNS:
            values = pd.to_numeric(group.get(col), errors="coerce")
            row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bar(summary, task, metric, ylabel, output_path):
    data = summary if task == "overall" else summary[summary["task_name"] == task]
    if task == "overall":
        data = data.groupby("ablation_mode").agg({f"{metric}_mean": "mean", f"{metric}_std": "mean"}).reset_index()
    if data.empty:
        return
    plt.figure(figsize=(10, 5))
    x = np.arange(len(data))
    plt.bar(x, data[f"{metric}_mean"], yerr=data[f"{metric}_std"], capsize=4)
    plt.xticks(x, data["ablation_mode"], rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{task} {ylabel}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_convergence(result_root, task, output_path):
    plt.figure(figsize=(10, 6))
    task_root = result_root if task == "overall" else os.path.join(result_root, task)
    modes = sorted([d for d in os.listdir(task_root) if os.path.isdir(os.path.join(task_root, d))]) if os.path.exists(task_root) else []
    for mode in modes:
        curves = []
        mode_root = os.path.join(task_root, mode)
        for seed_dir in os.listdir(mode_root) if os.path.exists(mode_root) else []:
            csv_path = os.path.join(mode_root, seed_dir, "convergence.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if {"oracle_calls", "top10"}.issubset(df.columns):
                    curves.append(df[["oracle_calls", "top10"]])
        if not curves:
            continue
        grid = sorted(set().union(*[set(c["oracle_calls"]) for c in curves]))
        vals = []
        for c in curves:
            vals.append(np.interp(grid, c["oracle_calls"], c["top10"]))
        arr = np.array(vals)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        plt.plot(grid, mean, label=mode)
        plt.fill_between(grid, mean - std, mean + std, alpha=0.15)
    plt.xlabel("Oracle Calls")
    plt.ylabel("Top-10 Score")
    plt.title(f"{task} convergence top10")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    result_root = os.path.abspath(args.result_root)
    df = collect_metrics(result_root)
    os.makedirs(result_root, exist_ok=True)
    by_seed = os.path.join(result_root, "single_ablation_summary_by_seed.csv")
    df.to_csv(by_seed, index=False)
    summary = mean_std(df) if not df.empty else pd.DataFrame()
    summary_path = os.path.join(result_root, "single_ablation_summary_mean_std.csv")
    summary.to_csv(summary_path, index=False)
    if summary.empty:
        print("No metrics found.")
        return
    tasks = sorted(summary["task_name"].dropna().unique().tolist()) + ["overall"]
    for task in tasks:
        suffix = task.lower()
        plot_bar(summary, task, "auc_top_10", "AUC Top-10", os.path.join(result_root, f"{suffix}_single_ablation_auc_top10_bar.png"))
        plot_bar(summary, task, "mean_pairwise_tanimoto_distance", "Mean Pairwise Tanimoto Distance", os.path.join(result_root, f"{suffix}_single_ablation_diversity_bar.png"))
        plot_bar(summary, task, "scaffold_entropy", "Scaffold Entropy", os.path.join(result_root, f"{suffix}_single_ablation_scaffold_entropy_bar.png"))
        plot_convergence(result_root, task, os.path.join(result_root, f"{suffix}_single_ablation_convergence_top10.png"))
    plot_bar(summary, "overall", "auc_top_10", "AUC Top-10", os.path.join(result_root, "single_ablation_auc_top10_bar.png"))
    plot_convergence(result_root, "overall", os.path.join(result_root, "single_ablation_convergence_top10.png"))
    plot_bar(summary, "overall", "mean_pairwise_tanimoto_distance", "Mean Pairwise Tanimoto Distance", os.path.join(result_root, "single_ablation_diversity_bar.png"))
    plot_bar(summary, "overall", "scaffold_entropy", "Scaffold Entropy", os.path.join(result_root, "single_ablation_scaffold_entropy_bar.png"))
    print(f"Saved summaries to {result_root}")


if __name__ == "__main__":
    main()
