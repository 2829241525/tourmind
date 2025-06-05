#!/bin/bash

# 自动Git推送服务启动脚本
# 用于在后台运行auto_git_push.py

# 配置区域
PROJECT_DIR="/home/maxon/disk2/roomMatch/room_match"
PYTHON_SCRIPT="auto_git_push.py"
PID_FILE="$PROJECT_DIR/auto_git_push.pid"
LOG_FILE="$PROJECT_DIR/auto_git_push.log"

# 切换到项目目录
cd "$PROJECT_DIR" || {
    echo "错误：无法切换到项目目录 $PROJECT_DIR"
    exit 1
}

# 激活conda环境
source /home/maxon/anaconda3/bin/activate room_match

# 函数：启动服务
start_service() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "自动推送服务已在运行，PID: $(cat "$PID_FILE")"
        return 1
    fi
    
    echo "启动自动Git推送服务..."
    # 使用conda环境的绝对路径
    nohup /home/maxon/anaconda3/envs/room_match/bin/python "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "服务已启动，PID: $(cat "$PID_FILE")"
    echo "日志文件: $LOG_FILE"
}

# 函数：停止服务
stop_service() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "停止自动推送服务，PID: $PID"
            kill "$PID"
            rm -f "$PID_FILE"
            echo "服务已停止"
        else
            echo "服务未运行"
            rm -f "$PID_FILE"
        fi
    else
        echo "PID文件不存在，服务可能未运行"
    fi
}

# 函数：检查服务状态
check_status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "自动推送服务正在运行，PID: $(cat "$PID_FILE")"
        return 0
    else
        echo "自动推送服务未运行"
        return 1
    fi
}

# 函数：重启服务
restart_service() {
    stop_service
    sleep 2
    start_service
}

# 函数：查看日志
view_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo "=== 最近的日志 ==="
        tail -50 "$LOG_FILE"
    else
        echo "日志文件不存在"
    fi
}

# 主逻辑
case "$1" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        check_status
        ;;
    logs)
        view_logs
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "  start   - 启动自动推送服务"
        echo "  stop    - 停止自动推送服务"
        echo "  restart - 重启自动推送服务"
        echo "  status  - 检查服务状态"
        echo "  logs    - 查看最近的日志"
        exit 1
        ;;
esac 