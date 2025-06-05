#!/bin/bash

# 强化学习训练脚本
# 使用方法: ./run_rl_train.sh [start|stop|restart] [配置文件路径]
# 例如:
#   ./run_rl_train.sh start config/config_rl.json  # 启动训练
#   ./run_rl_train.sh stop                        # 停止训练
#   ./run_rl_train.sh restart config/config_rl.json # 重启训练

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
PID_FILE="${LOG_DIR}/train_rl.pid"

# 设置CUDA可见设备（如果有多个GPU，可以指定使用哪一个）
export CUDA_VISIBLE_DEVICES=0

# 设置PyTorch的线程数
export OMP_NUM_THREADS=4

# 获取操作类型，默认为start
ACTION="${1:-start}"
shift 2>/dev/null || true

# 获取配置文件路径，默认为config/config_rl.json
CONFIG_PATH="${1:-config/config_rl.json}"

# 显示系统信息
show_system_info() {
  echo "系统信息："
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
  echo "Python路径: $(which python)"
}

# 启动训练
start_training() {
  echo "使用配置文件：${CONFIG_PATH}"
  
  # 检查是否已有训练进程在运行
  if [ -f "${PID_FILE}" ] && ps -p $(cat "${PID_FILE}") > /dev/null; then
    echo "训练进程已在运行中 (PID: $(cat ${PID_FILE}))，如需重启请使用 restart 命令"
    return 1
  fi
  
  show_system_info
  
  # 运行训练脚本
  echo "开始强化学习训练..."
  nohup python train_rl.py --config "${CONFIG_PATH}" --seed 42 > "${LOG_DIR}/train_rl.log" 2>&1 &
  
  # 记录进程ID
  echo $! > "${PID_FILE}"
  echo "训练进程已启动 (PID: $(cat ${PID_FILE}))"
  echo "日志文件: ${LOG_DIR}/train_rl.log"
}

# 停止训练
stop_training() {
  if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if ps -p ${PID} > /dev/null; then
      echo "正在停止训练进程 (PID: ${PID})..."
      kill ${PID}
      sleep 2
      
      # 检查进程是否已经停止，如果没有则强制终止
      if ps -p ${PID} > /dev/null; then
        echo "进程未响应，正在强制终止..."
        kill -9 ${PID}
      fi
      
      echo "训练进程已停止"
    else
      echo "没有找到运行中的训练进程"
    fi
    rm -f "${PID_FILE}"
  else
    echo "PID文件不存在，没有运行中的训练进程"
  fi
}

# 重启训练
restart_training() {
  echo "正在重启训练..."
  stop_training
  start_training
}

# 根据操作类型执行相应的功能
case "${ACTION}" in
  start)
    start_training
    ;;
  stop)
    stop_training
    ;;
  restart)
    restart_training
    ;;
  *)
    echo "未知的操作: ${ACTION}"
    echo "用法: ./run_rl_train.sh [start|stop|restart] [配置文件路径]"
    exit 1
    ;;
esac

# 检查最后的操作结果
if [ $? -eq 0 ]; then
  echo "操作成功完成！"
else
  echo "操作失败，请检查日志文件。"
fi 