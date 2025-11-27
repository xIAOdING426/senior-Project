@echo off
REM Windows 环境设置脚本
REM 使用方法: 双击运行或在命令行中执行 setup.bat

echo 🚀 开始设置 DiffuSeg 项目环境...

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python 已安装

REM 创建虚拟环境
echo.
echo 📦 创建虚拟环境...
if exist venv (
    echo ⚠️  虚拟环境已存在，跳过创建
) else (
    python -m venv venv
    echo ✅ 虚拟环境创建成功
)

REM 激活虚拟环境并升级 pip
echo.
echo ⬆️  升级 pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

REM 安装依赖
echo.
echo 📥 安装项目依赖...
python -m pip install -r requirements.txt

REM 验证安装
echo.
echo 🔍 验证安装...
python -c "import torch; import torchvision; import diffusers; import PIL; import tqdm; import numpy; print('✅ 所有依赖安装成功！')" || (
    echo ❌ 依赖验证失败
    pause
    exit /b 1
)

echo.
echo ✅ 环境设置完成！
echo.
echo 📝 使用说明:
echo    1. 激活虚拟环境: venv\Scripts\activate
echo    2. 开始训练: cd diffusion ^&^& python train_ddpm.py
echo    3. 退出虚拟环境: deactivate
echo.
pause

