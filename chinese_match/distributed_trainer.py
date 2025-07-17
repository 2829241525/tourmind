# -*- coding: utf-8 -*-
"""
分布式训练模块
支持多GPU并行训练
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import logging
from typing import Optional, List, Dict, Any
import json

logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, master_addr: str = "localhost", master_port: str = "12355"):
    """
    初始化分布式训练环境
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        master_addr: 主节点地址
        master_port: 主节点端口
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # 初始化进程组
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # 设置当前设备
    torch.cuda.set_device(rank)
    
    logger.info(f"进程 {rank}/{world_size} 初始化完成，使用GPU {rank}")


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_available_gpus() -> List[int]:
    """
    获取可用的GPU列表
    
    Returns:
        可用GPU的索引列表
    """
    if not torch.cuda.is_available():
        return []
    
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        try:
            # 尝试分配少量内存来测试GPU是否可用
            torch.cuda.set_device(i)
            test_tensor = torch.zeros(1, device=f'cuda:{i}')
            del test_tensor
            available_gpus.append(i)
        except Exception as e:
            logger.warning(f"GPU {i} 不可用: {e}")
    
    return available_gpus


def select_gpus(gpu_config: Optional[str] = None, max_gpus: Optional[int] = None) -> List[int]:
    """
    根据配置选择GPU
    
    Args:
        gpu_config: GPU配置，可以是:
            - "auto": 自动选择所有可用GPU
            - "0,1,2": 指定GPU索引
            - "0-2": 指定GPU范围
        max_gpus: 最大使用GPU数量
    
    Returns:
        选中的GPU索引列表
    """
    available_gpus = get_available_gpus()
    
    if not available_gpus:
        logger.warning("没有可用的GPU")
        return []
    
    if gpu_config is None or gpu_config == "auto":
        selected_gpus = available_gpus
    elif "," in gpu_config:
        # 指定具体GPU索引
        selected_gpus = [int(g.strip()) for g in gpu_config.split(",") if g.strip().isdigit()]
    elif "-" in gpu_config:
        # 指定GPU范围
        start, end = map(int, gpu_config.split("-"))
        selected_gpus = list(range(start, end + 1))
    else:
        # 单个GPU
        selected_gpus = [int(gpu_config)] if gpu_config.isdigit() else []
    
    # 过滤出可用的GPU
    selected_gpus = [g for g in selected_gpus if g in available_gpus]
    
    # 限制最大GPU数量
    if max_gpus is not None:
        selected_gpus = selected_gpus[:max_gpus]
    
    if not selected_gpus:
        logger.warning("没有可用的GPU，将使用CPU训练")
        return []
    
    logger.info(f"选择GPU: {selected_gpus}")
    return selected_gpus


class DistributedTrainer:
    """分布式训练器包装类"""
    
    def __init__(self, base_trainer, gpu_config: str = "auto", max_gpus: Optional[int] = None):
        """
        初始化分布式训练器
        
        Args:
            base_trainer: 基础训练器实例
            gpu_config: GPU配置
            max_gpus: 最大GPU数量
        """
        self.base_trainer = base_trainer
        self.gpu_config = gpu_config
        self.max_gpus = max_gpus
        self.selected_gpus = select_gpus(gpu_config, max_gpus)
        self.world_size = len(self.selected_gpus)
        
        if self.world_size == 0:
            logger.warning("没有可用的GPU，将使用单GPU训练")
            self.distributed = False
        elif self.world_size == 1:
            logger.info("只有1个GPU，使用单GPU训练")
            self.distributed = False
        else:
            logger.info(f"启用分布式训练，使用 {self.world_size} 个GPU")
            self.distributed = True
    
    def setup_model_for_distributed(self, model, rank: int):
        """
        为分布式训练设置模型
        
        Args:
            model: 模型实例
            rank: 当前进程rank
        """
        if not self.distributed:
            return model
        
        # 将模型移动到指定GPU
        device = torch.device(f'cuda:{rank}')
        model = model.to(device)
        
        # 包装为DDP模型
        model = DDP(model, device_ids=[rank], output_device=rank)
        
        return model
    
    def setup_dataloader_for_distributed(self, dataset, batch_size: int, shuffle: bool = True, 
                                       num_workers: int = 4, rank: int = 0):
        """
        为分布式训练设置数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            rank: 当前进程rank
        
        Returns:
            分布式数据加载器
        """
        if self.distributed:
            # 使用DistributedSampler
            sampler = DistributedSampler(
                dataset, 
                num_replicas=self.world_size, 
                rank=rank,
                shuffle=shuffle
            )
            
            # 调整批次大小（每个GPU的批次大小）
            per_gpu_batch_size = batch_size // self.world_size
            
            dataloader = DataLoader(
                dataset,
                batch_size=per_gpu_batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            # 单GPU训练
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )
        
        return dataloader
    
    def train_distributed(self, train_func, *args, **kwargs):
        """
        启动分布式训练
        
        Args:
            train_func: 训练函数
            *args, **kwargs: 传递给训练函数的参数
        """
        if not self.distributed:
            # 单GPU训练
            return train_func(*args, **kwargs)
        
        # 启动多进程训练
        mp.spawn(
            self._train_worker,
            args=(train_func, args, kwargs),
            nprocs=self.world_size,
            join=True
        )
    
    def _train_worker(self, rank: int, train_func, args, kwargs):
        """
        训练工作进程
        
        Args:
            rank: 进程rank
            train_func: 训练函数
            args: 位置参数
            kwargs: 关键字参数
        """
        try:
            # 设置分布式环境
            setup_distributed(rank, self.world_size)
            
            # 设置当前GPU
            torch.cuda.set_device(rank)
            
            # 调用训练函数
            train_func(rank=rank, world_size=self.world_size, *args, **kwargs)
            
        except Exception as e:
            logger.error(f"进程 {rank} 训练失败: {e}")
            raise
        finally:
            # 清理分布式环境
            cleanup_distributed()


def create_distributed_trainer(base_trainer, gpu_config: str = "auto", max_gpus: Optional[int] = None):
    """
    创建分布式训练器
    
    Args:
        base_trainer: 基础训练器实例
        gpu_config: GPU配置
        max_gpus: 最大GPU数量
    
    Returns:
        分布式训练器实例
    """
    return DistributedTrainer(base_trainer, gpu_config, max_gpus)


def get_distributed_config() -> Dict[str, Any]:
    """
    获取分布式训练配置
    
    Returns:
        分布式配置字典
    """
    return {
        "gpu_config": "auto",  # 自动选择GPU
        "max_gpus": None,      # 不限制最大GPU数量
        "master_addr": "localhost",
        "master_port": "12355",
        "backend": "nccl"
    }


def is_distributed_available() -> bool:
    """
    检查是否可以进行分布式训练
    
    Returns:
        是否可以进行分布式训练
    """
    available_gpus = get_available_gpus()
    return len(available_gpus) > 1 and torch.cuda.is_available()


def print_distributed_info():
    """打印分布式训练信息"""
    available_gpus = get_available_gpus()
    logger.info(f"可用GPU数量: {len(available_gpus)}")
    logger.info(f"可用GPU索引: {available_gpus}")
    logger.info(f"分布式训练可用: {is_distributed_available()}")
    
    if available_gpus:
        for gpu_id in available_gpus:
            props = torch.cuda.get_device_properties(gpu_id)
            memory_gb = props.total_memory / 1024**3
            logger.info(f"GPU {gpu_id}: {props.name}, 内存: {memory_gb:.1f}GB") 