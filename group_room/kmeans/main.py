import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from models import GroupRequest, GroupResult, GroupResponse, RoomDescription
from clustering import perform_clustering, config as clustering_config
from text_processor import text_processor, config as text_config
import logging
import uvicorn
from typing import List, Optional, Dict
from datetime import datetime
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 全局配置
config = {
    'host': '0.0.0.0',
    'port': 8000,
    'log_level': 'info',
    'reload': True,
    'workers': 4,
    'cache_dir': '/home/maxon/disk2/roomMatch/room_match/group_room/cache',
    'max_batch_size': 1000,
    'min_batch_size': 2,
}

# 确保日志配置只被设置一次
logger = logging.getLogger(__name__)

# 创建缓存目录
Path(config['cache_dir']).mkdir(parents=True, exist_ok=True)

# 创建线程池
executor = ThreadPoolExecutor(max_workers=config['workers'])

app = FastAPI(
    title="房型分组服务",
    description="使用文本相似度和KMeans算法对房型进行分组",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_room_batch(
    rooms: List[RoomDescription],
    n_clusters: Optional[int] = None,
    min_cluster_size: Optional[int] = None
) -> Dict:
    """处理一批房型数据"""
    try:
        # 提取文本描述并向量化
        descriptions = [room.description for room in rooms]
        features = text_processor.process_texts(descriptions)

        # 执行聚类
        labels, n_clusters, cluster_info = perform_clustering(
            features, n_clusters, min_cluster_size
        )

        # 构建分组结果
        groups = []
        for room, label, similarities in zip(
            rooms, labels, cluster_info['centroid_similarities']
        ):
            similarity_score = similarities[label]
            groups.append(
                GroupResult(
                    room_id=room.room_id,
                    description=room.description,
                    group_id=int(label),
                    similarity_score=float(similarity_score)
                )
            )

        return {
            'groups': groups,
            'n_clusters': n_clusters,
            'group_statistics': cluster_info['metrics'],
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"批处理过程中出错: {str(e)}")
        raise


@app.post("/group", response_model=GroupResponse)
async def group_rooms(request: GroupRequest, background_tasks: BackgroundTasks):
    """对房型进行分组"""
    try:
        # 输入验证
        if len(request.rooms) > config['max_batch_size']:
            raise HTTPException(
                status_code=400,
                detail=f"房型数量超过限制 ({config['max_batch_size']})"
            )

        if len(request.rooms) < config['min_batch_size']:
            raise HTTPException(
                status_code=400,
                detail=f"房型数量不足 ({config['min_batch_size']})"
            )

        # 异步处理分组
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            process_room_batch,
            request.rooms,
            request.n_clusters,
            request.min_cluster_size
        )

        # 后台保存结果
        background_tasks.add_task(save_result_to_cache, result)

        return GroupResponse(**result)
    except Exception as e:
        logger.error(f"分组过程中出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """获取服务配置信息"""
    return {
        'clustering': clustering_config,
        'text_processing': text_config,
        'service': {k: v for k, v in config.items() if k != 'cache_dir'}
    }


@app.get("/stats")
async def get_stats():
    """获取服务统计信息"""
    cache_dir = Path(config['cache_dir'])
    cache_files = list(cache_dir.glob('*.json'))

    return {
        'total_requests': len(cache_files),
        'cache_size': sum(f.stat().st_size for f in cache_files),
        'last_request': max((f.stat().st_mtime for f in cache_files), default=0),
        'model_info': {
            'name': text_config['model_name'],
            'device': text_config['device']
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 测试文本处理
        test_text = ["test room"]
        features = text_processor.process_texts(test_text)

        return {
            "status": "healthy",
            "components": {
                "text_processor": "ok",
                "model": "ok",
                "cache": "ok"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def save_result_to_cache(result: Dict):
    """保存分组结果到缓存"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = Path(config['cache_dir']) / f"result_{timestamp}.json"

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存到缓存: {cache_file}")
        clean_old_cache_files()
    except Exception as e:
        logger.error(f"保存缓存失败: {str(e)}")


def clean_old_cache_files(max_files: int = 100):
    """清理旧的缓存文件"""
    try:
        cache_dir = Path(config['cache_dir'])
        cache_files = sorted(
            cache_dir.glob('*.json'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        # 保留最新的文件
        for f in cache_files[max_files:]:
            f.unlink()
            logger.info(f"已删除旧缓存文件: {f}")
    except Exception as e:
        logger.error(f"清理缓存失败: {str(e)}")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, config['log_level'].upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

    uvicorn.run(
        "main:app",
        host=config['host'],
        port=config['port'],
        log_level=config['log_level'],
        reload=config['reload'],
        workers=1 if config['reload'] else config['workers']
    )
