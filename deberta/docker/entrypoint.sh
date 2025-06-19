#!/bin/bash
set -e

# 打印当前工作目录和可用资源
echo "当前工作目录: $(pwd)"
echo "可用GPU:"
nvidia-smi || echo "GPU不可用，使用CPU模式运行"

# 创建必要的目录
mkdir -p /app/logs

# 打印环境信息
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 启动API服务
echo "启动房间匹配API服务..."
exec python -u /app/api_server_concurrent_fix.py 