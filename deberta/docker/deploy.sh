#!/bin/bash
# 房间匹配API服务部署脚本 - 无需docker-compose版本

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 容器和镜像名称
IMAGE_NAME="room-match-api:latest"
CONTAINER_NAME="room-match-api"
PORT=13001

# 显示帮助信息
show_help() {
    echo -e "${GREEN}房间匹配API服务部署脚本 (纯Docker版本)${NC}"
    echo
    echo "用法: $0 [命令]"
    echo
    echo "命令:"
    echo "  start       构建并启动API服务"
    echo "  stop        停止API服务"
    echo "  restart     重启API服务"
    echo "  logs        查看API服务日志"
    echo "  status      查看API服务状态"
    echo "  reload      重新加载模型"
    echo "  build       仅构建镜像"
    echo "  help        显示此帮助信息"
    echo
    echo "示例:"
    echo "  $0 start    # 构建并启动API服务"
    echo "  $0 logs     # 查看服务日志"
}

# 检查docker是否已安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}错误: docker未安装！${NC}"
        echo "请先安装docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
}

# 构建镜像
build_image() {
    echo -e "${GREEN}正在构建房间匹配API服务镜像...${NC}"
    
    # 检查本地镜像
    if ! docker images | grep -q "recommend.*latest"; then
        echo -e "${RED}错误：本地没有找到recommend:latest基础镜像！${NC}"
        return 1
    fi
    
    # 构建API镜像
    echo -e "${YELLOW}构建API服务镜像...${NC}"
    if docker build -t ${IMAGE_NAME} -f docker/Dockerfile .; then
        echo -e "${GREEN}房间匹配API服务镜像构建成功！${NC}"
        return 0
    else
        echo -e "${RED}房间匹配API服务镜像构建失败！${NC}"
        return 1
    fi
}

# 启动服务
start_service() {
    echo -e "${GREEN}正在启动房间匹配API服务...${NC}"
    
    # 检查容器是否已经存在
    if docker ps -a | grep -q ${CONTAINER_NAME}; then
        echo -e "${YELLOW}容器已存在，将先停止并删除...${NC}"
        docker stop ${CONTAINER_NAME} >/dev/null 2>&1
        docker rm ${CONTAINER_NAME} >/dev/null 2>&1
    fi
    
    # 检查镜像是否存在
    if ! docker images | grep -q "${IMAGE_NAME//:*/}.*${IMAGE_NAME//*:/}"; then
        echo -e "${YELLOW}镜像 ${IMAGE_NAME} 不存在，将先构建镜像...${NC}"
        build_image
        if [ $? -ne 0 ]; then
            echo -e "${RED}镜像构建失败，无法启动服务${NC}"
            exit 1
        fi
    fi
    
    # 检测GPU是否可用
    if command -v nvidia-smi &> /dev/null; then
        GPU_FLAG="--gpus all"
        echo -e "${GREEN}检测到NVIDIA GPU，将启用GPU支持${NC}"
    else
        GPU_FLAG=""
        echo -e "${YELLOW}未检测到NVIDIA GPU，将使用CPU模式运行${NC}"
    fi
    
    # 创建必要的目录
    mkdir -p logs data
    
    # 启动容器
    echo -e "${YELLOW}正在启动容器...${NC}"
    docker run -d \
        --name ${CONTAINER_NAME} \
        --restart unless-stopped \
        -p ${PORT}:13001 \
        -v $(pwd)/docker/logs:/app/logs \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/checkpoints_cross_entropy_side_fix:/app/checkpoints_cloud_result_nosplct \
        -e TZ=Asia/Shanghai \
        --ulimit nofile=65536:65536 \
        ${GPU_FLAG} \
        ${IMAGE_NAME}
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}容器启动失败！${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}等待服务启动...${NC}"
    
    # 等待容器启动并检查状态
    for i in {1..30}; do
        if ! docker ps | grep -q ${CONTAINER_NAME}; then
            echo -e "${RED}服务启动失败，请检查日志${NC}"
            docker logs ${CONTAINER_NAME} 2>/dev/null || echo "容器不存在"
            return 1
        fi
        
        echo -e "${GREEN}房间匹配API服务已成功启动！${NC}"
        echo "可通过以下方式访问服务："
        echo "  - API文档: http://localhost:${PORT}/docs"
        return 0
        
        echo -n "."
        sleep 2
    done
    
    echo -e "\n${YELLOW}服务启动超时，但容器已在运行中。您可以使用以下命令查看状态和日志：${NC}"
    echo "  1. 查看容器状态: $0 status"
    echo "  2. 查看容器日志: $0 logs"
}

# 停止服务
stop_service() {
    echo -e "${YELLOW}正在停止房间匹配API服务...${NC}"
    
    if docker ps | grep -q ${CONTAINER_NAME}; then
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
        echo -e "${GREEN}服务已停止并删除容器${NC}"
    else
        echo -e "${YELLOW}服务未运行${NC}"
    fi
}

# 查看服务日志
view_logs() {
    echo -e "${GREEN}房间匹配API服务日志:${NC}"
    
    if docker ps | grep -q ${CONTAINER_NAME}; then
        docker logs --tail=100 -f ${CONTAINER_NAME}
    else
        echo -e "${YELLOW}服务未运行，无法查看日志${NC}"
    fi
}

# 查看服务状态
check_status() {
    echo -e "${GREEN}房间匹配API服务状态:${NC}"
    
    if docker ps | grep -q ${CONTAINER_NAME}; then
        docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}"
        
        echo -e "\n${GREEN}API状态:${NC}"
        curl -s http://localhost:${PORT}/docs || echo -e "${RED}无法连接到API服务${NC}"
    else
        echo -e "${YELLOW}服务未运行${NC}"
        
        # 检查镜像是否存在
        if docker images | grep -q ${IMAGE_NAME//:*/}; then
            echo -e "${GREEN}镜像已构建: ${IMAGE_NAME}${NC}"
        else
            echo -e "${YELLOW}镜像未构建: ${IMAGE_NAME}${NC}"
        fi
    fi
}

# 重载模型
reload_model() {
    echo -e "${GREEN}正在重新加载模型...${NC}"
    
    if docker ps | grep -q ${CONTAINER_NAME}; then
        curl -X POST "http://localhost:${PORT}/reload" || echo -e "${RED}无法连接到API服务或重载接口不可用${NC}"
    else
        echo -e "${YELLOW}服务未运行，无法重载模型${NC}"
    fi
}

# 重启服务
restart_service() {
    echo -e "${GREEN}正在重启房间匹配API服务...${NC}"
    
    stop_service
    start_service
}

# 主函数
main() {
    check_docker
    
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
        logs)
            view_logs
            ;;
        status)
            check_status
            ;;
        reload)
            reload_model
            ;;
        build)
            build_image
            ;;
        *)
            show_help
            ;;
    esac
}

main "$@" 