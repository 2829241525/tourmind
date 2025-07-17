# -*- coding: utf-8 -*-
"""
分布式训练启动脚本
支持多GPU并行训练
"""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distributed_trainer import setup_distributed, cleanup_distributed, select_gpus, print_distributed_info
from trainer import DeBERTaTrainer

# 设置tokenizer并行性
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(
            os.path.dirname(__file__), 'logs', 'distributed_train.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_worker(rank: int, world_size: int, config_path: str, config_override: dict = None):
    """
    训练工作进程
    
    Args:
        rank: 进程rank
        world_size: 总进程数
        config_path: 配置文件路径
        config_override: 配置覆盖参数
    """
    try:
        # 设置分布式环境
        setup_distributed(rank, world_size)
        
        # 创建训练器
        trainer = DeBERTaTrainer(
            config_path,
            config_override=config_override,
            enable_distributed=True
        )
        
        # 开始训练
        trainer.train(rank=rank, world_size=world_size)
        
    except Exception as e:
        logger.error(f"进程 {rank} 训练失败: {e}")
        raise
    finally:
        # 清理分布式环境
        cleanup_distributed()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分布式训练DeBERTa模型')
    parser.add_argument(
        '--config', type=str, 
        default="/home/maxon/disk2/roomMatch/room_match/chinese_match/config/config_mdeberta_room_group.json", 
        help='配置文件路径'
    )
    parser.add_argument('--gpu-config', type=str, default='auto', 
                       help='GPU配置 (auto, 0,1,2, 0-2等)')
    parser.add_argument('--max-gpus', type=int, default=None, 
                       help='最大使用GPU数量')
    parser.add_argument('--master-addr', type=str, default='localhost', 
                       help='主节点地址')
    parser.add_argument('--master-port', type=str, default='12355', 
                       help='主节点端口')
    args = parser.parse_args()
    
    # 打印分布式训练信息
    print_distributed_info()
    
    # 选择GPU
    selected_gpus = select_gpus(args.gpu_config, args.max_gpus)
    
    if len(selected_gpus) == 0:
        logger.error("没有可用的GPU，无法启动分布式训练")
        return
    
    if len(selected_gpus) == 1:
        logger.info("只有1个GPU，将使用单GPU训练")
        # 创建单GPU训练器
        trainer = DeBERTaTrainer(args.config, enable_distributed=False)
        trainer.train()
        return
    
    # 设置环境变量
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    world_size = len(selected_gpus)
    logger.info(f"启动分布式训练，使用 {world_size} 个GPU: {selected_gpus}")
    
    # 启动多进程训练
    try:
        mp.spawn(
            train_worker,
            args=(world_size, args.config),
            nprocs=world_size,
            join=True
        )
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.exception(f"分布式训练失败: {str(e)}")


if __name__ == "__main__":
    main() 