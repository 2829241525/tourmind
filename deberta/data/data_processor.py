# -*- coding: utf-8 -*-
"""
数据处理脚本
支持数据集平衡和划分
"""

import pandas as pd
import argparse
import os
import logging
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
import sys
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class DataProcessor:
    """数据处理器"""

    def __init__(self, input_file: str):
        # 如果输入路径是相对路径，则相对于脚本目录
        if not os.path.isabs(input_file):
            input_file = os.path.join(SCRIPT_DIR, input_file)
        self.input_file = input_file
        self.df = None

        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

    def load_data(self) -> pd.DataFrame:
        """加载CSV文件"""
        logger.info(f"正在加载文件: {self.input_file}")
        try:
            # 检查文件大小
            file_size = os.path.getsize(self.input_file)
            logger.info(f"文件大小: {file_size / 1024 / 1024:.2f}MB")

            self.df = pd.read_csv(self.input_file)
            logger.info(f"成功加载 {len(self.df)} 行数据")
            return self.df
        except pd.errors.EmptyDataError:
            logger.error("文件为空")
            raise
        except pd.errors.ParserError:
            logger.error("文件格式错误，请确保是有效的CSV文件")
            raise
        except Exception as e:
            logger.error(f"加载文件失败: {str(e)}")
            raise

    def balance_by_label(self, label_column: str, pos_neg_ratio: Tuple[int, int], random_state: Optional[int] = None):
        """按标签比例整理数据，使正负样本达到指定比例

        Args:
            label_column: 标签列的名称
            pos_neg_ratio: 正负样本比例，格式为(正样本,负样本)，例如(1,4)表示正负比为1:4
            random_state: 随机种子
            
        Returns:
            平衡后的DataFrame
        """
        if self.df is None:
            self.load_data()

        try:
            # 检查标签列是否存在
            if label_column not in self.df.columns:
                raise ValueError(f"标签列 '{label_column}' 在数据集中不存在")

            # 将数据集按标签分为正负样本
            positive_samples = self.df[self.df[label_column] == 1]
            negative_samples = self.df[self.df[label_column] == 0]

            logger.info(f"原始数据集中正样本数量: {len(positive_samples)}, 负样本数量: {len(negative_samples)}")

            pos_ratio, neg_ratio = pos_neg_ratio
            
            # 确定最终的正负样本数量
            if len(positive_samples) / pos_ratio > len(negative_samples) / neg_ratio:
                # 负样本数量不足，以负样本为基准
                n_neg = len(negative_samples)
                n_pos = int(n_neg * pos_ratio / neg_ratio)
                logger.info(f"以负样本数量为基准，需要从 {len(positive_samples)} 个正样本中选择 {n_pos} 个")
                # 随机抽取正样本
                positive_samples = positive_samples.sample(n=n_pos, random_state=random_state)
            else:
                # 正样本数量不足，以正样本为基准
                n_pos = len(positive_samples)
                n_neg = int(n_pos * neg_ratio / pos_ratio)
                logger.info(f"以正样本数量为基准，需要从 {len(negative_samples)} 个负样本中选择 {n_neg} 个")
                # 随机抽取负样本
                negative_samples = negative_samples.sample(n=n_neg, random_state=random_state)

            # 合并正负样本
            balanced_df = pd.concat([positive_samples, negative_samples])
            
            # 打乱数据
            balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            # 输出最终数据分布情况
            total = len(balanced_df)
            n_pos_final = len(balanced_df[balanced_df[label_column] == 1])
            n_neg_final = len(balanced_df[balanced_df[label_column] == 0])
            
            logger.info(f"平衡后数据集分布情况:")
            logger.info(f"总数据量: {total}")
            logger.info(f"正样本: {n_pos_final} ({n_pos_final / total * 100:.2f}%)")
            logger.info(f"负样本: {n_neg_final} ({n_neg_final / total * 100:.2f}%)")
            logger.info(f"正负样本比例: 1:{n_neg_final / n_pos_final:.2f}")
            
            # 更新实例的dataframe为平衡后的数据
            self.df = balanced_df
            return balanced_df
            
        except Exception as e:
            logger.error(f"平衡数据集失败: {str(e)}")
            raise

    def split_datasets(self,
                       valid_size: float,
                       test_size: float,
                       train_file: str,
                       valid_file: str,
                       test_file: str,
                       random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """划分训练集、验证集和测试集"""
        if self.df is None:
            self.load_data()

        try:
            # 处理相对路径
            if not os.path.isabs(train_file):
                train_file = os.path.join(SCRIPT_DIR, train_file)
            if not os.path.isabs(valid_file):
                valid_file = os.path.join(SCRIPT_DIR, valid_file)
            if not os.path.isabs(test_file):
                test_file = os.path.join(SCRIPT_DIR, test_file)

            # 检查并创建输出目录
            for file_path in [train_file, valid_file, test_file]:
                output_dir = os.path.dirname(file_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    logger.info(f"创建输出目录: {output_dir}")

            # 验证划分比例
            if valid_size + test_size >= 1:
                raise ValueError("验证集和测试集的比例之和必须小于1")

            # 首先划分出测试集
            train_valid_df, test_df = train_test_split(
                self.df,
                test_size=test_size,
                random_state=random_state
            )

            # 然后从剩余数据中划分训练集和验证集
            # 计算验证集在剩余数据中的比例
            valid_ratio = valid_size / (1 - test_size)
            train_df, valid_df = train_test_split(
                train_valid_df,
                test_size=valid_ratio,
                random_state=random_state
            )

            # 保存训练集
            train_df.to_csv(train_file, index=False)
            logger.info(f"已保存训练集 ({len(train_df)} 行) 到: {train_file}")

            # 保存验证集
            valid_df.to_csv(valid_file, index=False)
            logger.info(f"已保存验证集 ({len(valid_df)} 行) 到: {valid_file}")

            # 保存测试集
            test_df.to_csv(test_file, index=False)
            logger.info(f"已保存测试集 ({len(test_df)} 行) 到: {test_file}")

            # 输出数据集分布情况
            total = len(self.df)
            logger.info(f"数据集分布情况:")
            logger.info(f"总数据量: {total}")
            logger.info(f"训练集: {len(train_df)} ({len(train_df) / total * 100:.2f}%)")
            logger.info(f"验证集: {len(valid_df)} ({len(valid_df) / total * 100:.2f}%)")
            logger.info(f"测试集: {len(test_df)} ({len(test_df) / total * 100:.2f}%)")

            return train_df, valid_df, test_df

        except Exception as e:
            logger.error(f"划分数据集失败: {str(e)}")
            raise
            
    def balance_and_split(self,
                          label_column: str,
                          pos_neg_ratio: Tuple[int, int],
                          valid_size: float,
                          test_size: float,
                          train_file: str,
                          valid_file: str,
                          test_file: str,
                          random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """先平衡数据集的正负样本比例，再划分训练集、验证集和测试集"""
        try:
            # 首先平衡样本
            logger.info("第一步：正在平衡数据集的正负样本比例...")
            self.balance_by_label(label_column, pos_neg_ratio, random_state)
            
            # 然后划分数据集
            logger.info("第二步：正在划分训练集、验证集和测试集...")
            return self.split_datasets(valid_size, test_size, train_file, valid_file, test_file, random_state)
            
        except Exception as e:
            logger.error(f"平衡并划分数据集失败: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='数据处理工具')

    # 基本参数
    parser.add_argument('--input', type=str, default='processed_merged_clean_ppt.csv',
                        help='输入CSV文件路径 (默认: processed_merged_cleaned_filtered.csv)')

    # 标签平衡参数
    parser.add_argument('--label-column', type=str, default='label',
                        help='标签列名 (默认: label)')
    parser.add_argument('--pos-ratio', type=int, default=1,
                        help='正样本比例 (默认: 1)')
    parser.add_argument('--neg-ratio', type=int, default=5,
                        help='负样本比例 (默认: 4)')

    # 划分模式参数
    parser.add_argument('--valid-size', type=float, default=0.1,
                        help='验证集比例 (默认: 0.1)')
    parser.add_argument('--test-size', type=float, default=0.1,
                        help='测试集比例 (默认: 0.1)')
    parser.add_argument('--train-file', type=str, default='processed_merged_clean_ppt_train.csv',
                        help='训练集输出路径 (默认: processed_merged_cleaned_filtered_train.csv)')
    parser.add_argument('--valid-file', type=str, default='processed_merged_clean_ppt_valid.csv',
                        help='验证集输出路径 (默认: processed_merged_cleaned_filtered_valid.csv)')    
    parser.add_argument('--test-file', type=str, default='processed_merged_clean_ppt_test.csv',
                        help='测试集输出路径 (默认: processed_merged_cleaned_filtered_test.csv)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='随机种子 (默认: 42)')

    args = parser.parse_args()

    try:
        # 创建处理器
        processor = DataProcessor(args.input)
        
        # 先平衡样本再划分数据集
        processor.balance_and_split(
            args.label_column,
            (args.pos_ratio, args.neg_ratio),
            args.valid_size,
            args.test_size,
            args.train_file,
            args.valid_file,
            args.test_file,
            args.random_state
        )
        
    except FileNotFoundError as e:
        logger.error(f"文件不存在: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
