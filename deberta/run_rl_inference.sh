#!/bin/bash

# 基于强化学习优化的文本匹配推理脚本
# 使用方法: ./run_rl_inference.sh [输入文件] [输出文件]
# 如果不提供参数，将使用默认配置

# 默认配置
DEFAULT_INPUT_FILE="./data/test.csv"
DEFAULT_OUTPUT_FILE="./results/predictions.csv"
DEFAULT_BATCH_SIZE=32

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

# 检查日志目录
mkdir -p "${SCRIPT_DIR}/logs"

# 设置模型路径
BASE_MODEL_PATH="${SCRIPT_DIR}/checkpoints_cross_entropy/best_model"
RL_MODEL_PATH="${SCRIPT_DIR}/checkpoints_rl/rl_model/best_precision"

# 处理参数
if [ "$#" -ge 1 ]; then
    INPUT_FILE="$1"
else
    echo "未提供输入文件路径，使用默认路径: ${DEFAULT_INPUT_FILE}"
    INPUT_FILE="${DEFAULT_INPUT_FILE}"
fi

if [ "$#" -ge 2 ]; then
    OUTPUT_FILE="$2"
else
    echo "未提供输出文件路径，使用默认路径: ${DEFAULT_OUTPUT_FILE}"
    OUTPUT_FILE="${DEFAULT_OUTPUT_FILE}"
fi

# 确保输出目录存在
mkdir -p "$(dirname "${OUTPUT_FILE}")"

# 检查输入文件是否存在
if [ ! -f "${INPUT_FILE}" ]; then
  echo "警告: 输入文件不存在: ${INPUT_FILE}"
  echo "请确认文件路径是否正确，或者先准备好输入文件。"
  exit 1
fi

# 检查模型是否存在
if [ ! -d "${BASE_MODEL_PATH}" ]; then
  echo "错误: 基础模型不存在: ${BASE_MODEL_PATH}"
  exit 1
fi

if [ ! -d "${RL_MODEL_PATH}" ]; then
  echo "错误: 强化学习模型不存在: ${RL_MODEL_PATH}"
  echo "请先运行 ./run_rl_train.sh 训练强化学习模型"
  exit 1
fi

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0

# 显示系统信息
echo "系统信息："
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Python路径: $(which python)"
echo "基础模型路径: ${BASE_MODEL_PATH}"
echo "强化学习模型路径: ${RL_MODEL_PATH}"
echo "输入文件: ${INPUT_FILE}"
echo "输出文件: ${OUTPUT_FILE}"
echo "批处理大小: ${DEFAULT_BATCH_SIZE}"

# 运行推理脚本
echo "开始运行推理..."
python inference_rl.py \
  --base-model "${BASE_MODEL_PATH}" \
  --rl-model "${RL_MODEL_PATH}" \
  --input "${INPUT_FILE}" \
  --output "${OUTPUT_FILE}" \
  --batch-size ${DEFAULT_BATCH_SIZE}

# 检查运行结果
if [ $? -eq 0 ]; then
  echo "推理成功完成！结果已保存至: ${OUTPUT_FILE}"
else
  echo "推理失败，请检查日志文件。"
  exit 1
fi

# 输出结果统计信息
if [ -f "${OUTPUT_FILE}" ]; then
  TOTAL_RECORDS=$(wc -l < "${OUTPUT_FILE}")
  TOTAL_RECORDS=$((TOTAL_RECORDS - 1))  # 减去标题行
  
  echo "结果统计："
  echo "总记录数: ${TOTAL_RECORDS}"
  
  # 如果有Python，可以提供更详细的统计
  if command -v python &> /dev/null; then
    echo "匹配统计："
    python -c "
import pandas as pd
df = pd.read_csv('${OUTPUT_FILE}')
match_count = df['match'].sum()
total = len(df)
print(f'  匹配样本数: {match_count} ({match_count/total:.2%})')
print(f'  不匹配样本数: {total-match_count} ({(total-match_count)/total:.2%})')

if 'label' in df.columns and df['label'].max() >= 0:
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true = df[df['label'] >= 0]['label'].values
    y_pred = df[df['label'] >= 0]['match'].astype(int).values
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f'  精确率: {precision:.4f}')
    print(f'  召回率: {recall:.4f}')
    print(f'  F1分数: {f1:.4f}')
"
  fi
fi 