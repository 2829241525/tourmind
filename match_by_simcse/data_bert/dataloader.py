# -*- coding: utf-8 -*-
"""
优化的数据加载器
支持多进程加载和预取功能
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Callable
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time

class PrefetchLoader:
    """
    预取数据加载器
    在GPU训练的同时预取下一批次数据
    """
    
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        
        self.loader_iter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
        
    def preload(self):
        """预加载下一批次数据"""
        try:
            self.next_batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return
            
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for k, v in self.next_batch.items()
            }
            
    def __iter__(self):
        return self
        
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
            
        self.preload()
        return batch
        
    def __len__(self):
        return len(self.loader)

class AsyncPrefetchLoader:
    """
    异步预取数据加载器
    使用线程池进行异步数据预处理
    """
    
    def __init__(self, 
                 loader: DataLoader,
                 device: torch.device,
                 prefetch_factor: int = 2,
                 transform_fn: Optional[Callable] = None):
        self.loader = loader
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.transform_fn = transform_fn
        
        # 创建预取队列
        self.queue = queue.Queue(maxsize=prefetch_factor)
        self.loader_iter = iter(loader)
        
        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=prefetch_factor)
        self.stop_event = threading.Event()
        
        # 启动预取线程
        self.prefetch_thread = threading.Thread(target=self._prefetch_data)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
    def _process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个批次数据"""
        # 应用自定义转换
        if self.transform_fn is not None:
            batch = self.transform_fn(batch)
            
        # 将数据移动到指定设备
        processed_batch = {
            k: v.to(self.device, non_blocking=True)
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        return processed_batch
        
    def _prefetch_data(self):
        """预取数据的后台线程"""
        try:
            while not self.stop_event.is_set():
                try:
                    batch = next(self.loader_iter)
                except StopIteration:
                    break
                    
                # 提交批处理任务到线程池
                future = self.executor.submit(self._process_batch, batch)
                
                # 等待处理完成并放入队列
                try:
                    processed_batch = future.result()
                    self.queue.put(processed_batch, timeout=1)
                except queue.Full:
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.1)
                    
        finally:
            # 放入结束标记
            self.queue.put(None)
            
    def __iter__(self):
        return self
        
    def __next__(self):
        batch = self.queue.get()
        if batch is None:
            raise StopIteration
        return batch
        
    def __len__(self):
        return len(self.loader)
        
    def __del__(self):
        self.stop_event.set()
        self.executor.shutdown(wait=False)

def create_optimized_loader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    device: torch.device = None,
    transform_fn: Optional[Callable] = None,
    async_loading: bool = True
) -> DataLoader:
    """
    创建优化的数据加载器
    
    Args:
        dataset: 数据集对象
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        prefetch_factor: 预取因子
        device: 计算设备
        transform_fn: 数据转换函数
        async_loading: 是否使用异步加载
        
    Returns:
        优化的数据加载器
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # 创建基础数据加载器
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 根据选择返回不同的预取加载器
    if async_loading:
        return AsyncPrefetchLoader(loader, device, prefetch_factor, transform_fn)
    else:
        return PrefetchLoader(loader, device) 