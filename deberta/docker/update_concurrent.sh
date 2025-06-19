#!/bin/bash
# 更新到并发版本的服务

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 显示脚本说明
echo -e "${GREEN}房间匹配API服务更新脚本 - 并发版本${NC}"
echo "此脚本将把服务更新到并发版本，使用api_server_concurrent_fix.py"

# 检查是否存在api_server_concurrent_fix.py
if [ ! -f "./api_server_concurrent_fix.py" ]; then
    echo -e "${RED}错误：api_server_concurrent_fix.py 文件不存在！${NC}"
    exit 1
fi

# 备份旧版本的服务文件（如果存在）
if [ -f "./api_server.py" ]; then
    echo -e "${YELLOW}备份原有的 api_server.py...${NC}"
    cp ./api_server.py ./api_server.py.bak
    echo -e "${GREEN}备份完成: api_server.py.bak${NC}"
fi

# 更新Dockerfile（如果已存在）
echo -e "${YELLOW}更新Dockerfile...${NC}"
sed -i 's/COPY api_server.py/COPY api_server_concurrent_fix.py/g' ./docker/Dockerfile

# 更新entrypoint.sh
echo -e "${YELLOW}更新entrypoint.sh...${NC}"
sed -i 's/api_server.py/api_server_concurrent_fix.py/g' ./docker/entrypoint.sh

# 更新requirements.txt，添加新的依赖项
echo -e "${YELLOW}更新requirements.txt...${NC}"
if ! grep -q "numpy" ./docker/requirements.txt; then
    echo "numpy>=1.22.0" >> ./docker/requirements.txt
fi
if ! grep -q "asyncio" ./docker/requirements.txt; then
    echo "asyncio>=3.4.3" >> ./docker/requirements.txt
fi


# 构建并启动新服务
echo -e "${GREEN}所有文件已更新，现在可以执行以下命令构建并启动新服务:${NC}"
echo -e "${YELLOW}./docker/deploy.sh build${NC} - 构建新的Docker镜像"
echo -e "${YELLOW}./docker/deploy.sh restart${NC} - 重启服务使用新版本"

# 询问是否立即构建并重启
read -p "是否立即构建并重启服务? (y/n): " answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    echo -e "${GREEN}开始构建镜像...${NC}"
    ./docker/deploy.sh build
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}构建成功，开始重启服务...${NC}"
        ./docker/deploy.sh restart
    else
        echo -e "${RED}构建失败，请检查错误并手动重试${NC}"
    fi
else
    echo -e "${YELLOW}已取消构建和重启，您可以稍后手动执行${NC}"
fi

echo -e "${GREEN}更新脚本执行完毕${NC}" 