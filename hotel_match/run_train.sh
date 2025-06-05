#!/bin/bash

# 设置工作目录
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORK_DIR"

# 配置
MAX_RESTARTS=5  # 最大重启次数
RESTART_DELAY=60  # 重启间隔（秒）
LOG_FILE="logs/train_monitor.log"
TRAIN_LOG="logs/train_$(date +%m%d).log"

# 确保日志目录存在
mkdir -p logs

# 记录日志的函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查程序是否在运行
is_running() {
    pgrep -f "python train.py" > /dev/null
}

# 启动训练程序
start_training() {
    log "启动训练程序..."
    nohup python train.py > "$TRAIN_LOG" 2>&1 &
    sleep 5
    if is_running; then
        log "训练程序已成功启动"
        return 0
    else
        log "训练程序启动失败"
        return 1
    fi
}

# 停止训练程序
stop_training() {
    log "正在停止训练程序..."
    pkill -f "python train.py"
    sleep 5
    if ! is_running; then
        log "训练程序已停止"
        return 0
    else
        log "训练程序停止失败"
        return 1
    fi
}

# 重启训练程序
restart_training() {
    stop_training
    start_training
}

# 监控和自动重启
monitor() {
    local restart_count=0
    
    log "开始监控训练程序..."
    
    while true; do
        if ! is_running; then
            restart_count=$((restart_count + 1))
            log "训练程序已停止，这是第 $restart_count 次重启"
            
            if [ $restart_count -gt $MAX_RESTARTS ]; then
                log "达到最大重启次数 ($MAX_RESTARTS)，停止监控"
                exit 1
            fi
            
            start_training
            sleep $RESTART_DELAY
        fi
        sleep 30
    done
}

# 根据命令行参数执行相应操作
case "$1" in
    start)
        start_training
        ;;
    stop)
        stop_training
        ;;
    restart)
        restart_training
        ;;
    monitor)
        monitor
        ;;
    *)
        echo "用法: $0 {start|stop|restart|monitor}"
        echo "  start   - 启动训练程序"
        echo "  stop    - 停止训练程序"
        echo "  restart - 重启训练程序"
        echo "  monitor - 启动监控模式（自动重启）"
        exit 1
        ;;
esac

exit 0 