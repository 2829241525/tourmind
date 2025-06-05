#!/bin/bash

# 获取运行中的API服务进程ID
PID=$(ps -ef | grep "python api_server.py" | grep -v grep | awk '{print $2}')

# 如果进程存在则杀死
if [ ! -z "$PID" ]; then
    echo "正在停止API服务 (PID: $PID)..."
    kill $PID
    sleep 2
fi

# 确保logs目录存在
mkdir -p logs

# 启动新的API服务
echo "正在启动新的API服务..."
cd $(dirname $0)
nohup python api_server.py > logs/api.log 2>&1 &

# 等待几秒确保服务启动
sleep 2

# 检查服务是否成功启动
NEW_PID=$(ps -ef | grep "python api_server.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$NEW_PID" ]; then
    echo "API服务已成功重启 (新PID: $NEW_PID)"
else
    echo "API服务启动失败，请检查日志文件"
fi 