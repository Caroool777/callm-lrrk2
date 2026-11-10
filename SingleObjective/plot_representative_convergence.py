"""Plot convergence curves for representative single-objective tasks.

The script reads the existing ``*_convergence_data.csv`` files produced by
``optimizer_base.py`` and creates both an overlay plot and a 2x3 panel plot.
It does not rerun any molecule optimization.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TASKS = (
    "qed",
    "jnk3",
    "gsk3b",
    "celecoxib_rediscovery",
    "mestranol_similarity",
    "isomers_c9h10n2o2pf2cl",
)

TASK_LABELS = {
    "qed": "QED",
    "drd2": "DRD2",
    "gsk3b": "GSK3B",
    "jnk3": "JNK3",
    "celecoxib_rediscovery": "Celecoxib Rediscovery",
    "mestranol_similarity": "Mestranol Similarity",
    "scaffold_hop": "Scaffold Hop",
    "isomers_c7h8n2o2": "Isomers C7H8N2O2",
    "isomers_c9h10n2o2pf2cl": "Isomers C9H10N2O2PF2Cl",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def convergence_path(result_dir: Path, task: str) -> Path:
    return result_dir / f"{task}_top_molecules" / f"{task}_convergence_data.csv"


def read_curve(path: Path) -> tuple[list[int], list[float]]:
    oracle_calls: list[int] = []
    avg_top10_scores: list[float] = []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"oracle_calls", "avg_top10_score"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")

        for row in reader:
            oracle_calls.append(int(float(row["oracle_calls"])))
            avg_top10_scores.append(float(row["avg_top10_score"]))

    if len(oracle_calls) < 2:
        raise ValueError(f"{path} has fewer than two convergence points")

    return oracle_calls, avg_top10_scores


def load_curves(result_dir: Path, tasks: tuple[str, ...]) -> dict[str, tuple[list[int], list[float]]]:
    curves: dict[str, tuple[list[int], list[float]]] = {}
    missing: list[Path] = []

    for task in tasks:
        path = convergence_path(result_dir, task)
        if not path.exists():
            missing.append(path)
            continue
        curves[task] = read_curve(path)

    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing convergence data:\n{missing_text}")

    return curves


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_overlay(curves: dict[str, tuple[list[int], list[float]]], output_path: Path) -> None:
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for color, (task, (calls, scores)) in zip(colors, curves.items()):
        label = TASK_LABELS.get(task, task)
        ax.plot(calls, scores, linewidth=2.2, color=color, label=label)

    style_axes(ax)
    ax.set_xlabel("Oracle calls", fontsize=12)
    ax.set_ylabel("Average top-10 score", fontsize=12)
    ax.set_title("Representative Single-Objective Convergence Curves", fontsize=14, weight="bold")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_panels(curves: dict[str, tuple[list[int], list[float]]], output_path: Path) -> None:
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True, sharey=True)

    for ax, color, (task, (calls, scores)) in zip(axes.flat, colors, curves.items()):
        label = TASK_LABELS.get(task, task)
        ax.plot(calls, scores, linewidth=2.1, color=color)
        style_axes(ax)
        ax.set_title(label, fontsize=11, weight="bold")
        ax.text(
            0.98,
            0.08,
            f"final={scores[-1]:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#111827",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Oracle calls", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Average top-10 score", fontsize=10)

    fig.suptitle("Representative Single-Objective Convergence Curves", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(curves: dict[str, tuple[list[int], list[float]]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "points", "final_oracle_calls", "final_avg_top10_score"])
        for task, (calls, scores) in curves.items():
            writer.writerow([task, len(calls), calls[-1], f"{scores[-1]:.6f}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot convergence curves for representative SingleObjective tasks."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Task names to plot. Defaults to six representative tasks.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=project_root() / "SingleResult",
        help="Directory containing *_top_molecules result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "SingleResult" / "representative_convergence",
        help="Directory for generated plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = tuple(args.tasks)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = load_curves(args.result_dir, tasks)
    plot_overlay(curves, output_dir / "representative_convergence_overlay.png")
    plot_panels(curves, output_dir / "representative_convergence_panels.png")
    write_summary(curves, output_dir / "representative_convergence_summary.csv")

    print(f"Plotted {len(curves)} tasks:")
    for task, (calls, scores) in curves.items():
        print(f"  {task}: {len(calls)} points, final {calls[-1]} calls, top-10 avg {scores[-1]:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
