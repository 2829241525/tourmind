#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置日志文件和PID文件
LOG_FILE="$SCRIPT_DIR/logs/api_server.log"
PID_FILE="$SCRIPT_DIR/api_server.pid"
PORT=8200

# 创建日志目录
mkdir -p "$SCRIPT_DIR/logs"

# 停止已运行的服务
if [ -f "$PID_FILE" ]; then
    echo "正在停止已运行的API服务..."
    PID=$(cat "$PID_FILE")
    kill -9 $PID 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "已停止PID为 $PID 的API服务"
    sleep 2
fi

# 激活conda环境（如果有）
# 注意：这里假设使用了conda环境，如果没有请修改或移除
if command -v conda >/dev/null 2>&1; then
    # 检查是否在base环境中，如果是则切换到指定环境
    CURRENT_ENV=$(conda info --envs | grep "*" | awk '{print $1}')
    TARGET_ENV="torch" # 替换为你的目标环境名称
    
    if [ "$CURRENT_ENV" = "base" ]; then
        echo "当前在conda base环境中，正在切换到 $TARGET_ENV 环境..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $TARGET_ENV
    fi
fi

# 启动API服务
echo "正在启动API服务，端口: $PORT..."
echo "工作目录: $SCRIPT_DIR"
python "$SCRIPT_DIR/api_server.py" > "$LOG_FILE" 2>&1 &

# 保存PID
echo $! > "$PID_FILE"
echo "API服务已启动，PID: $(cat "$PID_FILE")"
echo "日志文件: $LOG_FILE"

# 等待服务启动
echo "等待服务启动..."
sleep 3

# 检查服务是否成功启动
if ps -p $(cat "$PID_FILE") > /dev/null; then
    # 创建一个简单的使用说明文件
    cat > "$SCRIPT_DIR/HOW_TO_USE.txt" << EOF
====== 房间匹配API服务 ======

服务地址: http://localhost:$PORT

主要接口:
- POST /match/ - 房间匹配接口
- GET /health - 健康检查接口
- POST /reload - 重新加载模型
- GET /config - 查看当前配置

使用示例(使用curl):
curl -X POST "http://localhost:$PORT/match/" \\
    -H "Content-Type: application/json" \\
    -d '{
        "supplier_names": ["Supplier A"],
        "spl_room_names": ["Deluxe Room"],
        "spl_room_bed_names": ["King Bed"],
        "s_room_names": ["Standard Room"],
        "s_room_bed_names": ["Queen Bed"],
        "threshold": 0.5,
        "model": "cross_entropy"
    }'

查看服务状态:
curl http://localhost:$PORT/health

重新加载模型:
curl -X POST "http://localhost:$PORT/reload?model_type=cross_entropy"

查看配置:
curl http://localhost:$PORT/config

更多详情请访问: http://localhost:$PORT/docs
EOF
    
    echo "服务启动成功！使用说明已写入 $SCRIPT_DIR/HOW_TO_USE.txt"
    echo "API文档地址: http://localhost:$PORT/docs"
else
    echo "服务启动失败，请检查日志文件: $LOG_FILE"
    cat "$LOG_FILE"
fi 