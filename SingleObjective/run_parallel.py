"""
GPU 并行运行脚本 - 在多个 GPU 上同时运行不同任务

使用方法：
1. 运行所有任务（自动分配到可用GPU）:
   python run_parallel.py --all

2. 在指定GPU上运行指定任务:
   python run_parallel.py --tasks jnk3 qed gsk3b --gpus 0 1 2

3. 在单个GPU上运行单个任务:
   python run_parallel.py --tasks jnk3 --gpus 0

4. 查看所有可用任务:
   python run_parallel.py --list

5. 运行特定任务组:
   python run_parallel.py --group rediscovery --gpus 0 1 2
"""

import os
import sys
import argparse
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# 所有可用任务
ALL_TASKS = [
    'jnk3',
    'qed', 
    'gsk3b',
    'drd2',
    'deco_hop',
    'scaffold_hop',
    'valsartan_smarts',
    'celecoxib_rediscovery',
    'albuterol_similarity',
    'mestranol_similarity',
    'median1',
    'median2',
    'amlodipine_mpo',
    'osimertinib_mpo',
    'perindopril_mpo',
    'ranolazine_mpo',
    'sitagliptin_mpo',
    'zaleplon_mpo',
]

# 任务分组
TASK_GROUPS = {
    'rediscovery': ['celecoxib_rediscovery'],
    'similarity': ['albuterol_similarity', 'mestranol_similarity'],
    'kinase': ['jnk3', 'gsk3b'],
    'median': ['median1', 'median2'],
    'hop': ['deco_hop', 'scaffold_hop'],
    'core': ['jnk3', 'qed', 'gsk3b', 'drd2'],
    'pmo_mpo': ['amlodipine_mpo', 'osimertinib_mpo',
                'perindopril_mpo', 'ranolazine_mpo', 'sitagliptin_mpo',
                'zaleplon_mpo'],
}


def get_available_gpus():
    """获取可用的 GPU 数量"""
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def run_task(task_name, gpu_id, script_dir):
    """
    在指定 GPU 上运行单个任务
    
    :param task_name: 任务名称
    :param gpu_id: GPU ID
    :param script_dir: 脚本所在目录
    :return: (task_name, success, output)
    """
    repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    candidate_dirs = [script_dir, os.path.join(repo_root, "MultiObjective")]
    script_path = None
    for candidate_dir in candidate_dirs:
        candidate_path = os.path.join(candidate_dir, f"{task_name}.py")
        if os.path.exists(candidate_path):
            script_path = candidate_path
            break
    
    if script_path is None:
        return task_name, False, f"脚本不存在: {task_name}.py"
    
    # 设置环境变量指定 GPU
    env = os.environ.copy()
    if gpu_id == 'cpu':
        env.pop('CUDA_VISIBLE_DEVICES', None)
        device_arg = 'cpu'
    else:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device_arg = 'cuda:0'
    
    print(f"🚀 启动任务 [{task_name}] 在 GPU {gpu_id}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_path, '--device', device_arg],  # 因为 CUDA_VISIBLE_DEVICES 重映射，所以用 cuda:0
            cwd=os.path.dirname(script_path),
            env=env,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ 任务 [{task_name}] 完成 (GPU {gpu_id}, 耗时 {elapsed:.1f}s)")
            return task_name, True, result.stdout
        else:
            print(f"❌ 任务 [{task_name}] 失败 (GPU {gpu_id})")
            return task_name, False, result.stderr
    except Exception as e:
        return task_name, False, str(e)


def run_tasks_parallel(tasks, gpus, script_dir):
    """
    并行运行多个任务
    
    :param tasks: 任务列表
    :param gpus: GPU ID 列表
    :param script_dir: 脚本目录
    """
    if not gpus:
        print("❌ 没有可用的 GPU!")
        return
    
    num_gpus = len(gpus)
    print(f"\n📊 共 {len(tasks)} 个任务，使用 {num_gpus} 个 GPU")
    print(f"   GPUs: {gpus}")
    print(f"   Tasks: {tasks}\n")
    
    results = {}
    
    # 使用进程池并行执行
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        # 提交任务，循环分配 GPU
        futures = {}
        for i, task in enumerate(tasks):
            gpu_id = gpus[i % num_gpus]
            future = executor.submit(run_task, task, gpu_id, script_dir)
            futures[future] = task
        
        # 等待所有任务完成
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                name, success, output = future.result()
                results[name] = {'success': success, 'output': output}
            except Exception as e:
                results[task_name] = {'success': False, 'output': str(e)}
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("📊 运行汇总")
    print("=" * 60)
    
    success_count = sum(1 for r in results.values() if r['success'])
    fail_count = len(results) - success_count
    
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    
    if fail_count > 0:
        print("\n失败的任务:")
        for task, result in results.items():
            if not result['success']:
                print(f"  - {task}: {result['output'][:200]}...")


def run_tasks_sequential_on_gpus(tasks, gpus, script_dir):
    """
    在多个 GPU 上顺序但并行地运行任务
    每个 GPU 同时运行一个任务，完成后再运行下一个
    
    :param tasks: 任务列表
    :param gpus: GPU ID 列表
    :param script_dir: 脚本目录
    """
    if not gpus:
        print("❌ 没有可用的 GPU!")
        return
    
    num_gpus = len(gpus)
    print(f"\n📊 共 {len(tasks)} 个任务，使用 {num_gpus} 个 GPU 并行运行")
    print(f"   GPUs: {gpus}")
    print(f"   Tasks: {tasks}\n")
    
    # 将任务分配到各个 GPU
    gpu_tasks = {gpu: [] for gpu in gpus}
    for i, task in enumerate(tasks):
        gpu_id = gpus[i % num_gpus]
        gpu_tasks[gpu_id].append(task)
    
    print("任务分配:")
    for gpu_id, assigned_tasks in gpu_tasks.items():
        print(f"  GPU {gpu_id}: {assigned_tasks}")
    print()
    
    results = {}
    
    def run_gpu_tasks(gpu_id, task_list):
        """在单个 GPU 上顺序运行任务列表"""
        gpu_results = []
        for task in task_list:
            name, success, output = run_task(task, gpu_id, script_dir)
            gpu_results.append((name, success, output))
        return gpu_results
    
    # 每个 GPU 并行运行自己的任务队列
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {}
        for gpu_id, task_list in gpu_tasks.items():
            if task_list:
                future = executor.submit(run_gpu_tasks, gpu_id, task_list)
                futures[future] = gpu_id
        
        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                gpu_results = future.result()
                for name, success, output in gpu_results:
                    results[name] = {'success': success, 'output': output}
            except Exception as e:
                print(f"GPU {gpu_id} 执行出错: {e}")
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("📊 运行汇总")
    print("=" * 60)
    
    success_count = sum(1 for r in results.values() if r['success'])
    fail_count = len(results) - success_count
    
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    
    if fail_count > 0:
        print("\n失败的任务:")
        for task, result in results.items():
            if not result['success']:
                print(f"  - {task}: {result['output'][:200]}...")


def main():
    parser = argparse.ArgumentParser(
        description='GPU 并行运行分子优化任务',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_parallel.py --all                    # 运行所有任务
  python run_parallel.py --tasks jnk3 qed         # 运行指定任务
  python run_parallel.py --group rediscovery      # 运行 rediscovery 组任务
  python run_parallel.py --gpus 0 1 2 3 --all     # 使用指定 GPU
  python run_parallel.py --list                   # 列出所有任务
        """
    )
    
    parser.add_argument('--tasks', nargs='+', help='要运行的任务名称列表')
    parser.add_argument('--gpus', nargs='+', type=int, help='要使用的 GPU ID 列表')
    parser.add_argument('--all', action='store_true', help='运行所有任务')
    parser.add_argument('--group', type=str, choices=list(TASK_GROUPS.keys()), 
                       help='运行指定任务组')
    parser.add_argument('--list', action='store_true', help='列出所有可用任务')
    parser.add_argument('--sequential', action='store_true', 
                       help='每个GPU顺序执行分配给它的任务（默认所有任务同时启动）')
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 列出所有任务
    if args.list:
        print("\n📋 所有可用任务:")
        for task in ALL_TASKS:
            print(f"  - {task}")
        print("\n📦 任务组:")
        for group, tasks in TASK_GROUPS.items():
            print(f"  {group}: {tasks}")
        return
    
    # 确定要运行的任务
    if args.all:
        tasks = ALL_TASKS
    elif args.group:
        tasks = TASK_GROUPS[args.group]
    elif args.tasks:
        tasks = args.tasks
        # 验证任务名称
        for task in tasks:
            if task not in ALL_TASKS:
                print(f"⚠️ 未知任务: {task}")
                print(f"可用任务: {ALL_TASKS}")
                return
    else:
        parser.print_help()
        return
    
    # 确定要使用的 GPU
    available_gpus = get_available_gpus()
    if args.gpus:
        gpus = args.gpus
        for gpu in gpus:
            if gpu >= available_gpus:
                print(f"⚠️ GPU {gpu} 不存在，只有 {available_gpus} 个 GPU 可用")
                return
    else:
        gpus = list(range(available_gpus)) if available_gpus > 0 else []
    
    if not gpus:
        print("⚠️ 没有可用的 GPU，将使用 CPU 运行")
        gpus = ['cpu']
    
    print(f"🔧 检测到 {available_gpus} 个 GPU")
    
    # 运行任务
    if args.sequential:
        run_tasks_sequential_on_gpus(tasks, gpus, script_dir)
    else:
        run_tasks_parallel(tasks, gpus, script_dir)


if __name__ == "__main__":
    main()
