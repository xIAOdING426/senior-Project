# 快速开始指南

## 在新电脑上克隆项目后

### 方法1：使用自动安装脚本（推荐）

**macOS/Linux:**
```bash
git clone https://github.com/xIAOdING426/senior-Project.git
cd senior-Project
bash setup.sh
```

**Windows:**
```cmd
git clone https://github.com/xIAOdING426/senior-Project.git
cd senior-Project
setup.bat
```

### 方法2：手动安装

```bash
# 1. 创建虚拟环境
python3 -m venv venv

# 2. 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

## 准备数据

1. 下载 DRIVE 数据集
2. 将训练图像放到 `data/DRIVE/training/images/` 目录
3. 运行检查脚本验证数据：
```bash
python3 check_data.py
```

## 开始训练

```bash
# 确保虚拟环境已激活
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate     # Windows

# 进入代码目录
cd diffusion

# 开始训练
python3 train_ddpm.py
```

## 生成图像

训练完成后，使用采样脚本：

```bash
python3 sample_ddpm.py
```

记得在 `sample_ddpm.py` 中修改 `ckpt_path` 指向训练好的模型。

## 常用命令

```bash
# 激活虚拟环境
source venv/bin/activate

# 退出虚拟环境
deactivate

# 检查数据
python3 check_data.py

# 查看训练输出
ls outputs_ddpm/
```

