# -*- coding: utf-8 -*-

import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)

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

class RoomMatchTripletDataset(Dataset):
    """房间匹配数据集，用于训练集的三元组格式数据"""
    
    def __init__(self, file_paths: str, is_train: bool = True):
        """
        初始化数据集
        Args:
            file_paths: CSV文件路径，多个文件用逗号分隔
            is_train: 是否为训练集
        """
        # 读取所有数据文件
        dfs = []
        for file_path in file_paths.split(','):
            df = pd.read_csv(file_path.strip())
            dfs.append(df)
            logger.info(f"成功读取文件: {file_path}, 数据量: {len(df)}")
        
        # 合并所有数据
        self.data = pd.concat(dfs, ignore_index=True)
        self.is_train = is_train
        logger.info(f"合并后总数据量: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 获取锚点文本、正例文本和负例文本
        anchor_text = str(row['spl_room_text'])
        positive_text = str(row['s_room_text'])
        negative_text = str(row['negative'])
        
        return {
            'anchor_text': anchor_text,
            'positive_text': positive_text,
            'negative_text': negative_text
        }

class RoomMatchValidationDataset(Dataset):
    """房间匹配数据集，用于验证集的二分类格式数据"""
    
    def __init__(self, file_paths: str):
        """
        初始化数据集
        Args:
            file_paths: CSV文件路径，多个文件用逗号分隔
        """
        # 读取所有数据文件
        dfs = []
        for file_path in file_paths.split(','):
            df = pd.read_csv(file_path.strip())
            dfs.append(df)
            logger.info(f"成功读取验证文件: {file_path}, 数据量: {len(df)}")
        
        # 合并所有数据
        self.data = pd.concat(dfs, ignore_index=True)
        logger.info(f"验证集总数据量: {len(self.data)}")
        
        # 统计正负样本数量
        pos_count = (self.data['label'] == 1).sum()
        neg_count = (self.data['label'] == 0).sum()
        logger.info(f"验证集正样本数量: {pos_count}, 负样本数量: {neg_count}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        return {
            'text1': str(row['spl_room_text']),
            'text2': str(row['s_room_text']),
            'label': int(row['label'])
        } 