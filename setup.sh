#!/bin/bash
# 项目环境设置脚本
# 使用方法: bash setup.sh 或 chmod +x setup.sh && ./setup.sh

set -e  # 遇到错误立即退出

echo "🚀 开始设置 DiffuSeg 项目环境..."

# 检查 Python 版本
echo "📋 检查 Python 版本..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python 版本: $(python3 --version)"

# 创建虚拟环境
echo ""
echo "📦 创建虚拟环境..."
if [ -d "venv" ]; then
    echo "⚠️  虚拟环境已存在，跳过创建"
else
    python3 -m venv venv
    echo "✅ 虚拟环境创建成功"
fi

# 激活虚拟环境并升级 pip
echo ""
echo "⬆️  升级 pip..."
source venv/bin/activate
python3 -m pip install --upgrade pip --quiet

# 安装依赖
echo ""
echo "📥 安装项目依赖..."
python3 -m pip install -r requirements.txt

# 验证安装
echo ""
echo "🔍 验证安装..."
python3 -c "import torch; import torchvision; import diffusers; import PIL; import tqdm; import numpy; print('✅ 所有依赖安装成功！')" || {
    echo "❌ 依赖验证失败"
    exit 1
}

echo ""
echo "✅ 环境设置完成！"
echo ""
echo "📝 使用说明:"
echo "   1. 激活虚拟环境: source venv/bin/activate"
echo "   2. 开始训练: cd diffusion && python3 train_ddpm.py"
echo "   3. 退出虚拟环境: deactivate"
echo ""

