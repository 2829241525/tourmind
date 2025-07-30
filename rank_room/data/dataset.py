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
            logger.info(
                f"训练集正样本数量: {len(pos_samples)}, 负样本数量: {len(neg_samples)}")

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
            'spl_text': self.preprocess_text(row['text1']),
            's_text': self.preprocess_text(row['text2']),
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
        anchor_text = str(row['text1'])
        positive_text = str(row['text2'])
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
            'text1': str(row['text1']),
            'text2': str(row['text2']),
            'label': int(row['label'])
        }


class RoomComparisonDataset(Dataset):
    """房间比较数据集，用于三分类任务：HIGHER, LOWER, INCOMPARABLE"""

    def __init__(self, file_path: str):
        """
        初始化数据集
        Args:
            file_path: CSV文件路径
        """
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")

        # 读取CSV文件
        self.data = pd.read_csv(file_path)
        logger.info(f"成功读取文件: {file_path}, 数据量: {len(self.data)}")

        # 定义标签映射
        self.label_map = {
            'HIGHER': 0,      # room_a 优于 room_b
            'LOWER': 1,       # room_a 劣于 room_b
            'INCOMPARABLE': 2  # room_a 与 room_b 不可比
        }

        # 过滤掉EQUAL标签的数据（如果存在）
        before_filter = len(self.data)
        self.data = self.data[self.data['comparison_result'] != 'EQUAL']
        after_filter = len(self.data)
        if before_filter != after_filter:
            logger.info(f"过滤掉EQUAL标签数据: {before_filter - after_filter}个")

        # 转换标签为数字
        self.data['label'] = self.data['comparison_result'].map(self.label_map)

        # 检查是否有未知标签
        unknown_labels = self.data[self.data['label'].isna()]
        if len(unknown_labels) > 0:
            logger.warning(
                f"发现未知标签: {unknown_labels['comparison_result'].unique()}")
            # 移除未知标签的数据
            self.data = self.data.dropna(subset=['label'])

        # 转换标签为整数
        self.data['label'] = self.data['label'].astype(int)

        # 统计各类别数量
        label_counts = self.data['label'].value_counts().sort_index()
        logger.info("标签分布:")
        for label_id, count in label_counts.items():
            label_name = [k for k, v in self.label_map.items()
                          if v == label_id][0]
            logger.info(f"  {label_name} ({label_id}): {count}个")

        logger.info(f"最终数据量: {len(self.data)}")

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
            'text1': self.preprocess_text(str(row['room_a'])),  # 第一个房间描述
            'text2': self.preprocess_text(str(row['room_b'])),  # 第二个房间描述
            'label': int(row['label'])  # 0,1,2对应HIGHER,LOWER,INCOMPARABLE
        }

    def get_class_weights(self):
        """
        计算类别权重，用于处理类别不平衡
        """
        label_counts = self.data['label'].value_counts().sort_index()
        total_samples = len(self.data)
        num_classes = len(self.label_map)

        # 计算权重：总样本数 / (类别数 * 该类别样本数)
        weights = []
        for i in range(num_classes):
            if i in label_counts:
                weight = total_samples / (num_classes * label_counts[i])
            else:
                weight = 1.0  # 如果某个类别没有样本，设置默认权重
            weights.append(weight)

        logger.info(f"计算得到的类别权重: {weights}")
        return weights
