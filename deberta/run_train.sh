#!/bin/bash

# 设置工作目录
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORK_DIR"

# 日志配置
LOG_DIR="logs"
PID_FILE="${LOG_DIR}/train.pid"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 记录日志的函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查程序是否在运行
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null; then
            return 0
        fi
    fi
    return 1
}

# 获取程序状态
get_status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        echo "训练程序正在运行 (PID: $pid)"
        echo "日志文件: $(readlink -f $(find $LOG_DIR -name "train_*.log" -type f -newer $PID_FILE | head -n 1))"
    else
        echo "训练程序未运行"
    fi
}

# 启动训练程序
start() {
    if is_running; then
        echo "训练程序已在运行中"
        return 1
    fi

    log "启动训练程序..."
    
    # 使用默认配置启动训练
    local TRAIN_CMD="python train.py"
    log "执行命令: $TRAIN_CMD"
    
    # 使用nohup在后台运行
    nohup $TRAIN_CMD >> "$LOG_FILE" 2>&1 &
    local PID=$!
    echo $PID > "$PID_FILE"
    
    sleep 2
    if is_running; then
        log "训练程序已成功启动，PID: $PID"
        echo "日志文件: $(readlink -f $LOG_FILE)"
        return 0
    else
        log "训练程序启动失败"
        rm -f "$PID_FILE"
        return 1
    fi
}

# 停止训练程序
stop() {
    if ! is_running; then
        echo "训练程序未运行"
        return 0
    fi

    local pid=$(cat "$PID_FILE")
    log "正在停止训练程序 (PID: $pid)..."
    
    kill $pid
    sleep 2
    
    if is_running; then
        log "强制终止训练程序..."
        kill -9 $pid
        sleep 1
    fi
    
    rm -f "$PID_FILE"
    log "训练程序已停止"
}

# 重启训练程序
restart() {
    log "重启训练程序..."
    stop
    sleep 2
    start
}

# 显示帮助信息
show_help() {
    echo "用法: $0 {start|stop|restart|status}"
    echo "命令:"
    echo "  start   启动训练程序"
    echo "  stop    停止训练程序"
    echo "  restart 重启训练程序"
    echo "  status  查看程序状态"
    exit 1
}

# 主程序
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        get_status
        ;;
    *)
        show_help
        ;;
esac

exit 0 