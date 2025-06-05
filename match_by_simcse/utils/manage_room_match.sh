#!/bin/bash

# 设置环境变量
export CONDA_ENV="room_match"
export PROJECT_ROOT="/home/maxon/disk2/roomMatch/room_match"
export APP_FILE="room_all_country_evaluate_viewer.py"
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOG_FILE="$SCRIPT_DIR/room_match_app.log"
export PID_FILE="$SCRIPT_DIR/room_match_app.pid"
export PORT=8504

# 切换到项目目录
cd $PROJECT_ROOT

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# 函数：检查应用是否在运行
check_running() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null; then
            return 0  # 正在运行
        fi
    fi
    return 1  # 未运行
}

# 函数：停止应用
stop_app() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null; then
            echo "正在停止房型匹配分析工具..."
            kill $pid
            sleep 2
            if ps -p $pid > /dev/null; then
                echo "强制停止进程..."
                kill -9 $pid
            fi
        fi
        rm -f "$PID_FILE"
    fi
}

# 函数：启动应用
start_app() {
    echo "正在启动房型匹配分析工具..."
    cd $PROJECT_ROOT
    PYTHONPATH=$PROJECT_ROOT nohup streamlit run match_by_simcse/utils/$APP_FILE \
        --server.port $PORT \
        --server.address 0.0.0.0 \
        > "$LOG_FILE" 2>&1 &
    
    echo $! > "$PID_FILE"
    echo "应用已启动，PID: $(cat $PID_FILE)"
    echo "日志文件: $LOG_FILE"
    echo "访问地址: http://0.0.0.0:$PORT"
}

# 主要命令处理
case "$1" in
    start)
        if check_running; then
            echo "房型匹配分析工具已经在运行中 (PID: $(cat $PID_FILE))"
        else
            start_app
        fi
        ;;
    stop)
        if check_running; then
            stop_app
            echo "房型匹配分析工具已停止"
        else
            echo "房型匹配分析工具未在运行"
        fi
        ;;
    restart)
        if check_running; then
            stop_app
        fi
        start_app
        ;;
    status)
        if check_running; then
            echo "房型匹配分析工具正在运行 (PID: $(cat $PID_FILE))"
            echo "日志文件: $LOG_FILE"
            echo "访问地址: http://0.0.0.0:$PORT"
        else
            echo "房型匹配分析工具未在运行"
        fi
        ;;
    *)
        echo "使用方法: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac

exit 0 