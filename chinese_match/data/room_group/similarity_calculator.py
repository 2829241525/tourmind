#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用mDeBERTa-v3-base模型计算CSV文件中两列文本的embedding相似度
"""

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings('ignore')


class SimilarityCalculator:
    def __init__(self, model_path):
        """
        初始化相似度计算器

        Args:
            model_path: mDeBERTa模型路径
        """
        print("正在加载模型...")
        self.device = torch.device(
            'cuda:1' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("模型加载完成!")

    def get_sentence_embedding(self, text):
        """
        获取句子的embedding向量

        Args:
            text: 输入文本

        Returns:
            embedding向量
        """
        if pd.isna(text) or text == '':
            # 对于空文本，返回零向量
            return torch.zeros(768).to(self.device)

        # 分词和编码
        inputs = self.tokenizer(
            str(text),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        # 移动到对应设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 使用平均池化获取句子embedding（推荐方法）
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state

        # 计算平均池化
        sentence_embedding = torch.sum(
            token_embeddings * attention_mask.unsqueeze(-1),
            1
        ) / torch.clamp(attention_mask.sum(1, keepdim=True), min=1e-9)

        return sentence_embedding.squeeze()

    def calculate_similarity(self, text1, text2):
        """
        计算两个文本的余弦相似度

        Args:
            text1, text2: 要比较的两个文本

        Returns:
            相似度分数 (0-1之间)
        """
        # 获取两个句子的embedding
        emb1 = self.get_sentence_embedding(text1)
        emb2 = self.get_sentence_embedding(text2)

        # 计算余弦相似度
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return similarity.item()

    def process_csv_batch(self, df_chunk, col1_name, col2_name):
        """
        批量处理数据块

        Args:
            df_chunk: 数据块
            col1_name: 第一列名称
            col2_name: 第二列名称

        Returns:
            相似度列表
        """
        similarities = []

        for i, row in df_chunk.iterrows():
            text1 = row[col1_name]
            text2 = row[col2_name]

            try:
                similarity = self.calculate_similarity(text1, text2)
                similarities.append(round(similarity, 4))
            except Exception as e:
                print(f"处理第{i+1}行时出错: {e}")
                similarities.append(0.0)

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return similarities

    def process_csv(self, csv_path, col1_name, col2_name, output_path=None, batch_size=1000):
        """
        处理CSV文件，计算两列之间的相似度

        Args:
            csv_path: CSV文件路径
            col1_name: 第一列名称
            col2_name: 第二列名称
            output_path: 输出文件路径，如果为None则覆盖原文件
            batch_size: 批处理大小
        """
        print(f"正在读取CSV文件: {csv_path}")

        # 分块读取CSV文件
        chunk_iter = pd.read_csv(csv_path, chunksize=batch_size)

        # 获取第一个块来检查列名和获取总行数
        first_chunk = next(chunk_iter)
        print(f"列名: {list(first_chunk.columns)}")

        # 检查列是否存在
        if col1_name not in first_chunk.columns:
            raise ValueError(f"列 '{col1_name}' 不存在于CSV文件中")
        if col2_name not in first_chunk.columns:
            raise ValueError(f"列 '{col2_name}' 不存在于CSV文件中")

        # 获取总行数
        total_rows = sum(1 for line in open(csv_path)) - 1  # 减去标题行
        print(f"文件包含 {total_rows} 行数据")

        # 重新开始读取
        chunk_iter = pd.read_csv(csv_path, chunksize=batch_size)

        all_chunks = []
        all_similarities = []

        print("正在计算相似度...")

        # 创建进度条
        pbar = tqdm(total=total_rows, desc="处理进度")

        for chunk_idx, chunk in enumerate(chunk_iter):
            # 计算当前块的相似度
            similarities = self.process_csv_batch(chunk, col1_name, col2_name)

            # 添加相似度列到当前块
            chunk['similarity_score'] = similarities
            all_chunks.append(chunk)
            all_similarities.extend(similarities)

            # 更新进度条
            pbar.update(len(chunk))

            # 每处理10个批次输出一次进度信息
            if (chunk_idx + 1) % 10 == 0:
                current_avg = np.mean(similarities)
                print(
                    f"\n已处理 {(chunk_idx + 1) * batch_size} 行, 当前批次平均相似度: {current_avg:.4f}")

        pbar.close()

        # 合并所有块
        print("正在合并结果...")
        df_result = pd.concat(all_chunks, ignore_index=True)

        # 保存结果
        if output_path is None:
            output_path = csv_path.replace('.csv', '_with_similarity.csv')

        print(f"正在保存结果到: {output_path}")
        df_result.to_csv(output_path, index=False, encoding='utf-8')
        print(f"结果已保存!")

        # 显示统计信息
        print(f"\n相似度统计:")
        print(f"平均相似度: {np.mean(all_similarities):.4f}")
        print(f"最大相似度: {np.max(all_similarities):.4f}")
        print(f"最小相似度: {np.min(all_similarities):.4f}")
        print(f"标准差: {np.std(all_similarities):.4f}")

        # 显示前几行结果
        print(f"\n前5行结果预览:")
        display_cols = [col1_name, col2_name, 'similarity_score']
        print(df_result[display_cols].head())

        return df_result


def main():
    # 模型路径
    model_path = "/home/maxon/disk2/roomMatch/room_match/chinese_match/pretrained_models/mdeberta-v3-base"

    # CSV文件路径 - 更新为新的文件
    csv_path = "/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group/room_group_train_fix_clean_filtered_set_different.csv"

    # 列名 - 更新为新的列名
    col1_name = "spl_room_text"
    col2_name = "s_room_text"

    # 批处理大小（可以根据GPU内存调整）
    batch_size = 100  # 减小批次大小以处理大文件

    try:
        # 创建相似度计算器
        calculator = SimilarityCalculator(model_path)

        # 处理CSV文件
        result_df = calculator.process_csv(
            csv_path, col1_name, col2_name, batch_size=batch_size)

        print("\n处理完成！")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
