#!/bin/bash
# ============================================================
# CALLM 实验运行脚本
# 文化算法 + BioT5 分子优化框架
# ============================================================

# 配置
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${DEVICE:-cuda:0}"  # 默认使用 cuda:0，可通过环境变量修改
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "CALLM 分子优化实验"
echo "项目目录: $PROJECT_DIR"
echo "使用设备: $DEVICE"
echo "日志目录: $LOG_DIR"
echo "=============================================="

# ============================================================
# 单目标优化任务
# ============================================================

run_single_objective() {
    echo ""
    echo "========== 运行单目标优化任务 =========="
    
    cd "$PROJECT_DIR"
    
    # 定义单目标任务列表
    SINGLE_TASKS=(
        "qed"
        "jnk3"
        "gsk3b"
        "drd2"
        "celecoxib_rediscovery"
        "albuterol_similarity"
        "mestranol_similarity"
        "median1"
        "median2"
        "scaffold_hop"
        "deco_hop"
        "valsartan_smarts"
    )
    
    for task in "${SINGLE_TASKS[@]}"; do
        echo "---------- Running: $task ----------"
        python SingleObjective/${task}.py --device "$DEVICE"
    done
}

run_single_objective_nohup() {
    echo ""
    echo "========== 后台运行所有单目标任务 =========="
    
    cd "$PROJECT_DIR"
    LOG_FILE="$LOG_DIR/single_objective_${TIMESTAMP}.log"
    
    nohup bash -c "
        python SingleObjective/qed.py --device $DEVICE && \
        python SingleObjective/jnk3.py --device $DEVICE && \
        python SingleObjective/gsk3b.py --device $DEVICE && \
        python SingleObjective/drd2.py --device $DEVICE && \
        python SingleObjective/celecoxib_rediscovery.py --device $DEVICE && \
        python SingleObjective/albuterol_similarity.py --device $DEVICE && \
        python SingleObjective/mestranol_similarity.py --device $DEVICE && \
        python SingleObjective/median1.py --device $DEVICE && \
        python SingleObjective/median2.py --device $DEVICE && \
        python SingleObjective/scaffold_hop.py --device $DEVICE && \
        python SingleObjective/deco_hop.py --device $DEVICE && \
        python SingleObjective/valsartan_smarts.py --device $DEVICE
    " > "$LOG_FILE" 2>&1 &
    
    echo "任务已在后台启动，PID: $!"
    echo "日志文件: $LOG_FILE"
    echo "查看日志: tail -f $LOG_FILE"
}

# ============================================================
# 多目标优化任务 (MPO)
# ============================================================

run_multi_objective() {
    echo ""
    echo "========== 运行多目标优化任务 =========="
    
    cd "$PROJECT_DIR"
    
    # 定义多目标任务列表
    MPO_TASKS=(
        "sitagliptin_mpo"
        "zaleplon_mpo"
        "amlodipine_mpo"
        "osimertinib_mpo"
        "perindopril_mpo"
        "ranolazine_mpo"
    )
    
    for task in "${MPO_TASKS[@]}"; do
        echo "---------- Running: $task ----------"
        if [ "$task" == "zaleplon_mpo" ]; then
            python MultiObjective/${task}.py --device "$DEVICE" --use_tdc
        else
            python MultiObjective/${task}.py --device "$DEVICE"
        fi
    done
}

run_multi_objective_nohup() {
    echo ""
    echo "========== 后台运行所有多目标任务 =========="
    
    cd "$PROJECT_DIR"
    LOG_FILE="$LOG_DIR/multi_objective_${TIMESTAMP}.log"
    
    nohup bash -c "
        python MultiObjective/sitagliptin_mpo.py --device $DEVICE && \
        python MultiObjective/zaleplon_mpo.py --device $DEVICE --use_tdc && \
        python MultiObjective/amlodipine_mpo.py --device $DEVICE && \
        python MultiObjective/osimertinib_mpo.py --device $DEVICE && \
        python MultiObjective/perindopril_mpo.py --device $DEVICE && \
        python MultiObjective/ranolazine_mpo.py --device $DEVICE
    " > "$LOG_FILE" 2>&1 &
    
    echo "任务已在后台启动，PID: $!"
    echo "日志文件: $LOG_FILE"
    echo "查看日志: tail -f $LOG_FILE"
}

# ============================================================
# 运行所有任务
# ============================================================

run_all() {
    echo ""
    echo "========== 运行所有优化任务 =========="
    run_single_objective
    run_multi_objective
}

run_all_nohup() {
    echo ""
    echo "========== 后台运行所有任务 =========="
    
    cd "$PROJECT_DIR"
    LOG_FILE="$LOG_DIR/all_tasks_${TIMESTAMP}.log"
    
    nohup bash -c "
        # 单目标任务
        python SingleObjective/qed.py --device $DEVICE && \
        python SingleObjective/jnk3.py --device $DEVICE && \
        python SingleObjective/gsk3b.py --device $DEVICE && \
        python SingleObjective/drd2.py --device $DEVICE && \
        
        # 多目标任务
        python MultiObjective/sitagliptin_mpo.py --device $DEVICE && \
        python MultiObjective/zaleplon_mpo.py --device $DEVICE --use_tdc && \
        python MultiObjective/amlodipine_mpo.py --device $DEVICE && \
        python MultiObjective/osimertinib_mpo.py --device $DEVICE && \
        python MultiObjective/perindopril_mpo.py --device $DEVICE && \
        python MultiObjective/ranolazine_mpo.py --device $DEVICE
    " > "$LOG_FILE" 2>&1 &
    
    echo "所有任务已在后台启动，PID: $!"
    echo "日志文件: $LOG_FILE"
    echo "查看日志: tail -f $LOG_FILE"
}

# ============================================================
# 单独运行某个任务
# ============================================================

run_task() {
    TASK_NAME="$1"
    TASK_TYPE="$2"  # single 或 multi
    
    if [ -z "$TASK_NAME" ]; then
        echo "错误: 请指定任务名称"
        echo "用法: ./run_experiments.sh task <任务名> [single|multi]"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    
    if [ "$TASK_TYPE" == "multi" ]; then
        echo "运行多目标任务: $TASK_NAME"
        python MultiObjective/${TASK_NAME}.py --device "$DEVICE"
    else
        echo "运行单目标任务: $TASK_NAME"
        python SingleObjective/${TASK_NAME}.py --device "$DEVICE"
    fi
}

# ============================================================
# 工具函数
# ============================================================

check_status() {
    echo ""
    echo "========== 检查运行状态 =========="
    ps aux | grep -E "python.*(SingleObjective|MultiObjective)" | grep -v grep
}

show_logs() {
    echo ""
    echo "========== 最近的日志文件 =========="
    ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -5
}

# ============================================================
# 主函数
# ============================================================

show_help() {
    echo ""
    echo "用法: ./run_experiments.sh <命令> [选项]"
    echo ""
    echo "命令:"
    echo "  single        - 运行所有单目标优化任务"
    echo "  single-bg     - 后台运行所有单目标任务"
    echo "  multi         - 运行所有多目标优化任务"
    echo "  multi-bg      - 后台运行所有多目标任务"
    echo "  all           - 运行所有任务"
    echo "  all-bg        - 后台运行所有任务"
    echo "  task <名称>   - 运行指定任务"
    echo "  status        - 检查运行状态"
    echo "  logs          - 显示最近的日志文件"
    echo ""
    echo "环境变量:"
    echo "  DEVICE        - 指定GPU设备 (默认: cuda:0)"
    echo ""
    echo "示例:"
    echo "  ./run_experiments.sh multi-bg"
    echo "  DEVICE=cuda:1 ./run_experiments.sh multi-bg"
    echo "  ./run_experiments.sh task qed single"
    echo "  ./run_experiments.sh task sitagliptin_mpo multi"
    echo ""
}

case "$1" in
    single)
        run_single_objective
        ;;
    single-bg)
        run_single_objective_nohup
        ;;
    multi)
        run_multi_objective
        ;;
    multi-bg)
        run_multi_objective_nohup
        ;;
    all)
        run_all
        ;;
    all-bg)
        run_all_nohup
        ;;
    task)
        run_task "$2" "$3"
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h|*)
        show_help
        ;;
esac
