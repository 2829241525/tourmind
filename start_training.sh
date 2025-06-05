#!/bin/bash

# 定义基础变量
BASE_DIR=~/disk2/roomMatch/room_match
LOG_DIR=${BASE_DIR}/sequence_annotation/logs
SCRIPT_PATH=${BASE_DIR}/sequence_annotation/room_bed_extraction.py
OUTPUT_LOG=${LOG_DIR}/training_output.log
PID_FILE=${LOG_DIR}/training.pid

# 确保日志目录存在
mkdir -p ${LOG_DIR}

# 激活conda环境
source ~/anaconda3/bin/activate room_match

# 清空之前的日志文件
> ${OUTPUT_LOG}

# 输出当前环境信息
echo "=== 环境信息 ===" | tee -a ${OUTPUT_LOG}
echo "时间: $(date)" | tee -a ${OUTPUT_LOG}
echo "主机: $(hostname)" | tee -a ${OUTPUT_LOG}
echo "当前用户: $(whoami)" | tee -a ${OUTPUT_LOG}
echo "当前环境: $(conda info --envs | grep '*')" | tee -a ${OUTPUT_LOG}
echo "Python版本: $(python --version 2>&1)" | tee -a ${OUTPUT_LOG}
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')" | tee -a ${OUTPUT_LOG}
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q 'True'; then
  echo "CUDA版本: $(python -c 'import torch; print(torch.version.cuda)')" | tee -a ${OUTPUT_LOG}
  echo "GPU型号: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')" | tee -a ${OUTPUT_LOG}
fi
echo "================" | tee -a ${OUTPUT_LOG}
echo "" | tee -a ${OUTPUT_LOG}

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=1

# 设置PYTHONUNBUFFERED环境变量，确保Python输出不被缓冲
export PYTHONUNBUFFERED=1

# 输出启动信息
echo "启动训练任务..." | tee -a ${OUTPUT_LOG}
echo "日志输出到: ${OUTPUT_LOG}" | tee -a ${OUTPUT_LOG}
echo "" | tee -a ${OUTPUT_LOG}
echo "训练进程即将开始..." | tee -a ${OUTPUT_LOG}
echo "================" | tee -a ${OUTPUT_LOG}

# 启动训练脚本
nohup python -u ${SCRIPT_PATH} >> ${OUTPUT_LOG} 2>&1 &

# 保存进程ID
PID=$!
echo $PID > ${PID_FILE}
echo ""
echo "训练任务已在后台启动，进程ID: $PID"
echo "可以通过以下命令查看训练日志:"
echo "  tail -f ${OUTPUT_LOG}"
echo ""
echo "可以通过以下命令停止训练:"
echo "  kill \$(cat ${PID_FILE})"
echo ""
echo "5秒后自动显示日志输出..."
sleep 5
tail -f ${OUTPUT_LOG} 