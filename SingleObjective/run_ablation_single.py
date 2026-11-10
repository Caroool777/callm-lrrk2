import argparse
import csv
import json
import os
import subprocess
import sys


TASK_SCRIPTS = {
    "jnk3": "jnk3.py",
    "gsk3b": "gsk3b.py",
    "drd2": "drd2.py",
    "albuterol_similarity": "albuterol_similarity.py",
    "mestranol_similarity": "mestranol_similarity.py",
    "celecoxib_rediscovery": "celecoxib_rediscovery.py",
    "thiothixene_rediscovery": "thiothixene_rediscovery.py",
    "troglitazone_rediscovery": "troglitazone_rediscovery.py",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-objective CALLM ablations")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--ablation_modes", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--max_calls", type=int, default=10000)
    parser.add_argument("--gpus", nargs="*", default=[])
    parser.add_argument("--result_root", default=os.path.join("MultiResult", "ablation_single"))
    return parser.parse_args()


def load_metrics(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_dir = os.path.abspath(os.path.dirname(__file__))
    result_root = os.path.abspath(os.path.join(repo_root, args.result_root))
    os.makedirs(result_root, exist_ok=True)
    summary_rows = []
    run_index = 0

    for task in args.tasks:
        if task not in TASK_SCRIPTS:
            print(f"Warning: unsupported task {task}; recording failed rows.")
        for mode in args.ablation_modes:
            for seed in args.seeds:
                output_dir = os.path.join(result_root, task, mode, f"seed_{seed}")
                os.makedirs(output_dir, exist_ok=True)
                stdout_path = os.path.join(output_dir, "stdout.log")
                stderr_path = os.path.join(output_dir, "stderr.log")
                config_path = os.path.join(output_dir, "ablation_config.json")
                config = {
                    "task_name": task,
                    "ablation_mode": mode,
                    "seed": seed,
                    "max_oracle_calls": args.max_calls,
                    "output_dir": output_dir,
                    "output_image_dir": output_dir,
                }
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

                row = {
                    "task_name": task,
                    "ablation_mode": mode,
                    "seed": seed,
                    "status": "failed",
                    "output_dir": output_dir,
                    "stdout_log": stdout_path,
                    "stderr_log": stderr_path,
                }
                script_name = TASK_SCRIPTS.get(task)
                if not script_name:
                    row["error"] = "unsupported_task"
                    summary_rows.append(row)
                    continue

                env = os.environ.copy()
                env["CALLM_ABLATION_CONFIG"] = config_path
                cmd = [sys.executable, os.path.join(script_dir, script_name)]
                if args.gpus:
                    gpu = str(args.gpus[run_index % len(args.gpus)])
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                    cmd.extend(["--device", "cuda:0"])
                else:
                    print("No GPU specified; falling back to CPU/default device.")
                run_index += 1

                print(f"Running {task} mode={mode} seed={seed} -> {output_dir}")
                with open(stdout_path, "w", encoding="utf-8") as stdout_f, open(stderr_path, "w", encoding="utf-8") as stderr_f:
                    proc = subprocess.run(cmd, cwd=repo_root, env=env, stdout=stdout_f, stderr=stderr_f)
                if proc.returncode == 0:
                    metrics = load_metrics(os.path.join(output_dir, "metrics.json"))
                    row.update(metrics)
                    row["status"] = "success"
                else:
                    row["returncode"] = proc.returncode
                    row["error"] = f"returncode_{proc.returncode}"
                summary_rows.append(row)

    summary_path = os.path.join(result_root, "summary.csv")
    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
