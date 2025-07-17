#!/bin/bash

# 分布式训练启动脚本
# 使用方法: ./run_distributed_train.sh [配置文件路径] [GPU配置] [最大GPU数量]
# 例如:
#   ./run_distributed_train.sh config/config_mdeberta_room_group.json auto 4  # 自动选择最多4个GPU
#   ./run_distributed_train.sh config/config_mdeberta_room_group.json 0,1,2  # 使用GPU 0,1,2
#   ./run_distributed_train.sh config/config_mdeberta_room_group.json 0-2    # 使用GPU 0,1,2

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

# 检查并创建日志目录
LOG_DIR="logs"
if [ ! -d "${LOG_DIR}" ]; then
  mkdir -p "${LOG_DIR}"
  echo "创建日志目录：${LOG_DIR}"
fi

# 进程ID文件
PID_FILE="${LOG_DIR}/distributed_train.pid"

# 设置PyTorch的线程数
export OMP_NUM_THREADS=4

# 获取参数
CONFIG_PATH="${1:-config/config_mdeberta_room_group.json}"
GPU_CONFIG="${2:-auto}"
MAX_GPUS="${3:-}"

# 显示系统信息
show_system_info() {
  echo "系统信息："
  echo "配置文件: ${CONFIG_PATH}"
  echo "GPU配置: ${GPU_CONFIG}"
  if [ -n "${MAX_GPUS}" ]; then
    echo "最大GPU数量: ${MAX_GPUS}"
  fi
  echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
  echo "Python路径: $(which python)"
}

# 启动分布式训练
start_distributed_training() {
  echo "使用配置文件：${CONFIG_PATH}"
  echo "GPU配置：${GPU_CONFIG}"
  
  # 检查是否已有训练进程在运行
  if [ -f "${PID_FILE}" ] && ps -p $(cat "${PID_FILE}") > /dev/null; then
    echo "训练进程已在运行中 (PID: $(cat ${PID_FILE}))，如需重启请使用 restart 命令"
    return 1
  fi
  
  show_system_info
  
  # 构建命令
  CMD="python run_distributed_train.py --config ${CONFIG_PATH} --gpu-config ${GPU_CONFIG}"
  if [ -n "${MAX_GPUS}" ]; then
    CMD="${CMD} --max-gpus ${MAX_GPUS}"
  fi
  
  # 运行分布式训练脚本
  echo "开始分布式训练..."
  echo "执行命令: ${CMD}"
  nohup ${CMD} > "${LOG_DIR}/distributed_train.log" 2>&1 &
  
  # 记录进程ID
  echo $! > "${PID_FILE}"
  echo "分布式训练进程已启动 (PID: $(cat ${PID_FILE}))"
  echo "日志文件: ${LOG_DIR}/distributed_train.log"
}

# 停止训练
stop_training() {
  if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if ps -p ${PID} > /dev/null; then
      echo "停止训练进程 (PID: ${PID})..."
      kill ${PID}
      rm -f "${PID_FILE}"
      echo "训练进程已停止"
    else
      echo "训练进程不存在或已停止"
      rm -f "${PID_FILE}"
    fi
  else
    echo "未找到训练进程ID文件"
  fi
}

# 重启训练
restart_training() {
  echo "重启分布式训练..."
  stop_training
  sleep 2
  start_distributed_training
}

# 查看状态
show_status() {
  if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if ps -p ${PID} > /dev/null; then
      echo "分布式训练进程正在运行 (PID: ${PID})"
      echo "日志文件: ${LOG_DIR}/distributed_train.log"
    else
      echo "训练进程不存在或已停止"
      rm -f "${PID_FILE}"
    fi
  else
    echo "未找到训练进程"
  fi
}

# 查看日志
show_logs() {
  if [ -f "${LOG_DIR}/distributed_train.log" ]; then
    tail -f "${LOG_DIR}/distributed_train.log"
  else
    echo "日志文件不存在"
  fi
}

# 主逻辑
case "${4:-start}" in
  "start")
    start_distributed_training
    ;;
  "stop")
    stop_training
    ;;
  "restart")
    restart_training
    ;;
  "status")
    show_status
    ;;
  "logs")
    show_logs
    ;;
  *)
    echo "使用方法: $0 [配置文件] [GPU配置] [最大GPU数量] [命令]"
    echo "命令: start|stop|restart|status|logs"
    echo "示例:"
    echo "  $0 config/config_mdeberta_room_group.json auto 4 start"
    echo "  $0 config/config_mdeberta_room_group.json 0,1,2 start"
    echo "  $0 config/config_mdeberta_room_group.json 0-2 start"
    ;;
esac 