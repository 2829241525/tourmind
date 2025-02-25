# -*- coding: utf-8 -*-

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
import os
import logging
from typing import Dict, Any, List
from transformers import AutoTokenizer
import mmap
import pickle
from pathlib import Path
import re

logger = logging.getLogger(__name__)

# class CacheManager:
#     """缓存管理器，用于管理数据集的缓存"""
    
#     def __init__(self, cache_dir="./cache"):
#         self.cache_dir = Path(cache_dir)
#         self.cache_dir.mkdir(exist_ok=True)
        
#     def get_cache_path(self, data_path: str, prefix: str) -> Path:
#         """获取缓存文件路径"""
#         data_hash = str(hash(data_path))
#         return self.cache_dir / f"{prefix}_{data_hash}.pkl"
        
#     def load_cache(self, data_path: str, prefix: str) -> Any:
#         """加载缓存数据"""
#         cache_path = self.get_cache_path(data_path, prefix)
#         if cache_path.exists():
#             try:
#                 with cache_path.open('rb') as f:
#                     return pickle.load(f)
#             except Exception as e:
#                 logger.warning(f"加载缓存失败: {e}")
#         return None
        
#     def save_cache(self, data: Any, data_path: str, prefix: str):
#         """保存数据到缓存"""
#         cache_path = self.get_cache_path(data_path, prefix)
#         try:
#             with cache_path.open('wb') as f:
#                 pickle.dump(data, f)
#         except Exception as e:
#             logger.warning(f"保存缓存失败: {e}")

# class MemoryEfficientReader:
#     """内存效率型文件读取器"""
    
#     def __init__(self, file_path: str):
#         self.file_path = file_path
#         self.file_size = os.path.getsize(file_path)
        
#     def __enter__(self):
#         self.file = open(self.file_path, 'rb')
#         self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
#         return self
        
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.mmap.close()
#         self.file.close()
        
#     def read_lines(self):
#         """使用内存映射方式读取文件行"""
#         current_position = 0
#         while current_position < self.file_size:
#             line_end = self.mmap.find(b'\n', current_position)
#             if line_end == -1:
#                 line_end = self.file_size
#             line = self.mmap[current_position:line_end].decode('utf-8')
#             yield line.strip()
#             current_position = line_end + 1

# def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#     """
#     处理不同模式的批次数据
#     Args:
#         batch: 批次数据列表
#     Returns:
#         合并后的批次数据
#     """
#     # 确定批次的模式
#     mode = batch[0]['mode']
#     if not all(b['mode'] == mode for b in batch):
#         raise ValueError("批次中的样本模式不一致")
    
#     if mode == 'triplet':
#         return {
#             'mode': mode,
#             'anchor_input_ids': torch.stack([b['anchor_input_ids'] for b in batch]),
#             'anchor_attention_mask': torch.stack([b['anchor_attention_mask'] for b in batch]),
#             'positive_input_ids': torch.stack([b['positive_input_ids'] for b in batch]),
#             'positive_attention_mask': torch.stack([b['positive_attention_mask'] for b in batch]),
#             'negative_input_ids': torch.stack([b['negative_input_ids'] for b in batch]),
#             'negative_attention_mask': torch.stack([b['negative_attention_mask'] for b in batch])
#         }
#     else:  # infonce
#         return {
#             'mode': mode,
#             'input_ids1': torch.stack([b['input_ids1'] for b in batch]),
#             'attention_mask1': torch.stack([b['attention_mask1'] for b in batch]),
#             'input_ids2': torch.stack([b['input_ids2'] for b in batch]),
#             'attention_mask2': torch.stack([b['attention_mask2'] for b in batch]),
#             'label': torch.stack([b['label'] for b in batch])
#         }

# class DualModeDataset(Dataset):
#     """支持两种训练模式的数据集，带有缓存和内存优化"""
    
#     def __init__(self, triplet_data_path=None, infonce_data_path=None, tokenizer=None, 
#                  max_length=128, mode=None, use_cache=True):
#         if tokenizer is None:
#             self.tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
#         else:
#             self.tokenizer = tokenizer
            
#         self.max_length = max_length
#         self.mode = mode
#         self.use_cache = use_cache
#         self.cache_manager = CacheManager() if use_cache else None
        
#         # 加载两种模式的数据
#         self.triplet_data = self._load_triplet_data(triplet_data_path) if triplet_data_path else []
#         self.infonce_data = self._load_infonce_data(infonce_data_path) if infonce_data_path else []
        
#         logger.info(f"加载了 {len(self.triplet_data)} 条triplet数据")
#         logger.info(f"加载了 {len(self.infonce_data)} 条infonce数据")
        
#         # 验证数据加载
#         if not self.triplet_data and not self.infonce_data:
#             raise ValueError("至少需要一种模式的数据")
#         if mode == 'triplet' and not self.triplet_data:
#             raise ValueError("指定了triplet模式但没有triplet数据")
#         if mode == 'infonce' and not self.infonce_data:
#             raise ValueError("指定了infonce模式但没有infonce数据")
    
#     def _load_triplet_data(self, data_path):
#         """加载triplet数据，支持缓存和内存映射"""
#         if not os.path.exists(data_path):
#             raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
#         if self.use_cache:
#             cached_data = self.cache_manager.load_cache(data_path, "triplet")
#             if cached_data is not None:
#                 logger.info("从缓存加载triplet数据")
#                 return cached_data
                
#         triplet_data = []
#         with MemoryEfficientReader(data_path) as reader:
#             for line in reader.read_lines():
#                 if not line:
#                     continue
#                 anchor, positive, negative = line.split('\t')
#                 triplet_data.append({
#                     'anchor': anchor,
#                     'positive': positive,
#                     'negative': negative
#                 })
                
#         if self.use_cache:
#             self.cache_manager.save_cache(triplet_data, data_path, "triplet")
            
#         return triplet_data
    
#     def _load_infonce_data(self, data_path):
#         """加载InfoNCE数据，支持缓存和内存映射"""
#         if not os.path.exists(data_path):
#             raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
#         if self.use_cache:
#             cached_data = self.cache_manager.load_cache(data_path, "infonce")
#             if cached_data is not None:
#                 logger.info("从缓存加载infonce数据")
#                 return cached_data
                
#         infonce_data = []
#         with MemoryEfficientReader(data_path) as reader:
#             for line in reader.read_lines():
#                 if not line:
#                     continue
#                 query, positive = line.split('\t')
#                 infonce_data.append({
#                     'query': query,
#                     'positive': positive
#                 })
                
#         if self.use_cache:
#             self.cache_manager.save_cache(infonce_data, data_path, "infonce")
            
#         return infonce_data
    
#     def __len__(self):
#         if self.mode == 'triplet':
#             return len(self.triplet_data)
#         elif self.mode == 'infonce':
#             return len(self.infonce_data)
#         else:
#             return len(self.triplet_data) + len(self.infonce_data)
    
#     def __getitem__(self, idx):
#         # 根据模式选择数据
#         if self.mode == 'triplet' or (self.mode is None and idx < len(self.triplet_data)):
#             # Triplet模式
#             if self.mode is None:
#                 triplet_idx = idx
#             else:
#                 triplet_idx = idx % len(self.triplet_data)
#             data = self.triplet_data[triplet_idx]
            
#             # 编码三元组
#             anchor_encoding = self.tokenizer(
#                 data['anchor'],
#                 max_length=self.max_length,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt'
#             )
            
#             positive_encoding = self.tokenizer(
#                 data['positive'],
#                 max_length=self.max_length,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt'
#             )
            
#             negative_encoding = self.tokenizer(
#                 data['negative'],
#                 max_length=self.max_length,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt'
#             )
            
#             return {
#                 'mode': 'triplet',
#                 'anchor_input_ids': anchor_encoding['input_ids'].squeeze(0),
#                 'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(0),
#                 'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
#                 'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
#                 'negative_input_ids': negative_encoding['input_ids'].squeeze(0),
#                 'negative_attention_mask': negative_encoding['attention_mask'].squeeze(0)
#             }
#         else:
#             # InfoNCE模式
#             if self.mode is None:
#                 infonce_idx = idx - len(self.triplet_data)
#             else:
#                 infonce_idx = idx % len(self.infonce_data)
#             data = self.infonce_data[infonce_idx]
            
#             # 编码文本对
#             text1_encoding = self.tokenizer(
#                 data['query'],
#                 max_length=self.max_length,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt'
#             )
            
#             text2_encoding = self.tokenizer(
#                 data['positive'],
#                 max_length=self.max_length,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt'
#             )
            
#             return {
#                 'mode': 'infonce',
#                 'input_ids1': text1_encoding['input_ids'].squeeze(0),
#                 'attention_mask1': text1_encoding['attention_mask'].squeeze(0),
#                 'input_ids2': text2_encoding['input_ids'].squeeze(0),
#                 'attention_mask2': text2_encoding['attention_mask'].squeeze(0),
#                 'label': torch.tensor(1, dtype=torch.long)
#             }

class RoomMatchDataset(Dataset):
    """房型匹配数据集"""
    
    def __init__(self, csv_files: str, is_train: bool = True):
        """
        初始化数据集
        Args:
            csv_files: CSV文件路径，多个文件用逗号分隔
            is_train: 是否为训练集
        """
        # 分割并处理多个CSV文件路径
        file_paths = [path.strip() for path in csv_files.split(',')]
        
        # 读取并合并所有CSV文件
        dfs = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                continue
            df = pd.read_csv(file_path)
            dfs.append(df)
            logger.info(f"成功读取文件: {file_path}, 数据量: {len(df)}")
        
        if not dfs:
            raise ValueError("没有成功读取任何CSV文件")
        
        # 合并所有数据框
        self.data = pd.concat(dfs, ignore_index=True)
        logger.info(f"合并后总数据量: {len(self.data)}")
        
        if is_train:
            # 分别获取正负样本
            pos_samples = self.data[self.data['label'] == 1]
            neg_samples = self.data[self.data['label'] == 0]
            logger.info(f"训练集正样本数量: {len(pos_samples)}, 负样本数量: {len(neg_samples)}")

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        预处理文本：
        1. 将逗号、英文逗号、句号替换为空格
        2. 将多个连续空格替换为单个空格
        """
        # 替换标点符号为空格
        text = re.sub(r'[,，.]', ' ', text)
        # 将多个空格替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'spl_text': self.preprocess_text(row['spl_room_text']),
            's_text': self.preprocess_text(row['s_room_text']),
            'label': row['label']
        } 