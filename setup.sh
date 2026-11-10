#!/bin/bash
# ============================================================
# CALLM 项目环境配置脚本
# 文化算法 + BioT5 分子优化框架
# ============================================================

echo "=============================================="
echo "CALLM 项目环境配置"
echo "=============================================="

# 方式1：使用 Conda 创建环境（推荐）
setup_conda() {
    echo "使用 Conda 创建环境..."
    conda env create -f environment.yml
    echo "环境创建完成！使用以下命令激活："
    echo "  conda activate callm"
}

# 方式2：使用 pip 安装依赖
setup_pip() {
    echo "使用 pip 安装依赖..."
    
    # 创建虚拟环境
    python -m venv venv
    source venv/bin/activate
    
    # 升级 pip
    pip install --upgrade pip
    
    # 安装 PyTorch (CUDA 11.8)
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    
    # 安装其他依赖
    pip install -r requirements.txt
    
    echo "依赖安装完成！"
}

# 检查 CUDA
check_cuda() {
    echo ""
    echo "检查 CUDA 环境..."
    python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU 数量: {torch.cuda.device_count()}')"
}

# 验证安装
verify_install() {
    echo ""
    echo "验证关键依赖..."
    python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except ImportError:
    print('❌ PyTorch 未安装')

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError:
    print('❌ Transformers 未安装')

try:
    from rdkit import Chem
    print('✅ RDKit 已安装')
except ImportError:
    print('❌ RDKit 未安装')

try:
    import selfies
    print(f'✅ SELFIES: {selfies.__version__}')
except ImportError:
    print('❌ SELFIES 未安装')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except ImportError:
    print('❌ NumPy 未安装')

try:
    import pandas as pd
    print(f'✅ Pandas: {pd.__version__}')
except ImportError:
    print('❌ Pandas 未安装')
"
}

# 主函数
main() {
    case "$1" in
        conda)
            setup_conda
            ;;
        pip)
            setup_pip
            ;;
        check)
            check_cuda
            verify_install
            ;;
        *)
            echo "用法: ./setup.sh [conda|pip|check]"
            echo "  conda - 使用 Conda 创建环境（推荐）"
            echo "  pip   - 使用 pip 安装依赖"
            echo "  check - 检查环境配置"
            ;;
    esac
}

main "$@"
