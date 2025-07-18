# -*- coding: utf-8 -*-
"""
房间分组相似度处理脚本
根据SHotelId和GroupId进行分组，计算组内房间的相似度并进行标记
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import logging
import torch.nn.functional as F
import random
from collections import defaultdict
import itertools

# ===================== 配置区域 =====================
# 路径配置
BASE_DIR = "/home/maxon/disk2/roomMatch/room_match/chinese_match"
MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints_group_result_fix/best_model')
INPUT_CSV_PATH = "/home/maxon/disk2/roomMatch/room_match/chinese_match/group_eval/v2_room_groupings_eval.csv"
OUTPUT_CSV_PATH = "/home/maxon/disk2/roomMatch/room_match/chinese_match/group_eval/v2_room_groupings_eval_processed.csv"

# 模型配置
MAX_LENGTH = 128
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
USE_FP16 = False

# 相似度阈值
SIMILARITY_THRESHOLD = 0.9

# 日志配置
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ===================== 日志设置 =====================
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)


class MDeBERTaPredictor:
    def __init__(self, model_path: str):
        """初始化预测器"""
        self.device = torch.device(DEVICE)
        logger.info(f"使用设备: {self.device}")

        # 加载模型和分词器
        try:
            if os.path.exists(model_path):
                # 加载配置文件
                config_path = os.path.join(
                    BASE_DIR, 'config', 'config_mdeberta.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    logger.info(f"成功加载配置文件: {config_path}")
                    self.temperature = self.config.get('temperature', 0.05)
                else:
                    logger.warning("未找到配置文件，使用默认配置")
                    self.temperature = 0.05

                # 加载分类器模型
                logger.info("使用mDeBERTa Classifier模型 (cross entropy loss)")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path).to(self.device)

                # 确保模型使用全精度计算
                if USE_FP16 is False:
                    self.model = self.model.float()
                    logger.info("已将模型转换为全精度(float32)计算")

                self.loss_type = 'cross_entropy'

                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                logger.info(f"成功从本地路径加载模型: {model_path}")

            else:
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise

        self.model.eval()

    def encode_text_pair(self, text1_list, text2_list, batch_size=BATCH_SIZE):
        """批量编码文本对，用于cross_entropy模式"""
        logits_list = []

        for i in range(0, len(text1_list), batch_size):
            batch_text1 = text1_list[i:i + batch_size]
            batch_text2 = text2_list[i:i + batch_size]

            # 对于分类器模型，使用文本对输入
            encoded = self.tokenizer(
                batch_text1,
                text_pair=batch_text2,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)

                # 获取正类的logits得分
                batch_logits = outputs.logits[:, 1]
                # 确保返回的是float32类型
                if batch_logits.dtype != torch.float32:
                    batch_logits = batch_logits.float()
                logits_list.append(batch_logits.cpu())

        return torch.cat(logits_list, dim=0)

    def calculate_similarity(self, source_embeddings, normalize_to_probability=True):
        """计算相似度，将logits转换为概率值"""
        # 获取logits作为相似度分数
        similarities = source_embeddings

        # 使用sigmoid函数将logits转换为0-1之间的概率值
        if normalize_to_probability:
            # 确保张量为float32类型
            if similarities.dtype != torch.float32:
                similarities = similarities.float()
            similarities = torch.sigmoid(similarities)

        return similarities

    def get_similarity_for_pair(self, text1: str, text2: str):
        """计算单对文本的相似度"""
        try:
            logits = self.encode_text_pair([text1], [text2])
            similarity = self.calculate_similarity(
                logits, normalize_to_probability=True)
            return float(similarity[0])
        except Exception as e:
            logger.error(f"计算相似度失败: {str(e)}")
            return 0.0


class GroupSimilarityProcessor:
    def __init__(self, model_path: str):
        """初始化分组相似度处理器"""
        self.predictor = MDeBERTaPredictor(model_path)

    def combine_room_text(self, room_name: str, bed_type_desc: str):
        """组合房间名称和床型描述"""
        room_name = str(room_name) if pd.notna(room_name) else ""
        bed_type_desc = str(bed_type_desc) if pd.notna(bed_type_desc) else ""
        return f"{room_name} {bed_type_desc}".strip().lower()

    def process_group_of_two(self, group_data):
        """处理包含2个房间的分组"""
        results = []

        # 获取两个房间的文本
        room1 = group_data.iloc[0]
        room2 = group_data.iloc[1]

        text1 = self.combine_room_text(room1['RoomName'], room1['BedTypeDesc'])
        text2 = self.combine_room_text(room2['RoomName'], room2['BedTypeDesc'])

        # 计算相似度
        similarity = self.predictor.get_similarity_for_pair(text1, text2)

        if similarity > SIMILARITY_THRESHOLD:
            # 相似度>0.9：标记列都标记为1，相似度随机一行存储
            mark_col = [1, 1]
            similarity_col = [similarity, ""] if random.choice([True, False]) else [
                "", similarity]
        else:
            # 相似度<0.9：标记列分别标记为1,2，相似度标记对应的相似度
            mark_col = [1, 2]
            similarity_col = [similarity, similarity]

        # 为每一行添加标记和相似度
        for i, (_, row) in enumerate(group_data.iterrows()):
            result_row = row.copy()
            result_row['标记列'] = mark_col[i]
            result_row['相似度列'] = similarity_col[i]
            results.append(result_row)

        return results

    def process_group_of_multiple(self, group_data):
        """处理包含3个或更多房间的分组"""
        results = []
        group_size = len(group_data)

        # 获取所有房间的文本
        room_texts = []
        for _, row in group_data.iterrows():
            text = self.combine_room_text(row['RoomName'], row['BedTypeDesc'])
            room_texts.append(text)

        # 计算两两之间的相似度
        similarity_matrix = {}
        all_pairs = list(itertools.combinations(range(group_size), 2))

        for i, j in all_pairs:
            similarity = self.predictor.get_similarity_for_pair(
                room_texts[i], room_texts[j])
            similarity_matrix[(i, j)] = similarity
            similarity_matrix[(j, i)] = similarity  # 对称

        # 判断所有相似度是否都大于阈值
        all_similar = all(
            sim > SIMILARITY_THRESHOLD for sim in similarity_matrix.values())

        if all_similar:
            # 所有两两相似度都>0.9：标记列统一标记为1
            mark_col = [1] * group_size
            similarity_col = []

            # 为每个房间生成相似度字符串（与其他房间的相似度）
            for i in range(group_size):
                other_similarities = []
                for j in range(group_size):
                    if i != j:
                        sim = similarity_matrix.get((i, j), 0.0)
                        other_similarities.append(f"{sim:.4f}")
                similarity_str = ",".join(other_similarities)
                similarity_col.append(similarity_str)
        else:
            # 有不相似的情况：需要进行聚类
            # 使用简单的聚类方法：基于相似度阈值
            clusters = []
            visited = set()

            for i in range(group_size):
                if i in visited:
                    continue

                cluster = [i]
                visited.add(i)

                for j in range(i + 1, group_size):
                    if j in visited:
                        continue

                    # 检查j是否与cluster中的所有元素都相似
                    similar_to_all = all(
                        similarity_matrix.get(
                            (k, j), 0.0) > SIMILARITY_THRESHOLD
                        for k in cluster
                    )

                    if similar_to_all:
                        cluster.append(j)
                        visited.add(j)

                clusters.append(cluster)

            # 分配标记
            mark_col = [0] * group_size
            similarity_col = [""] * group_size

            for cluster_id, cluster in enumerate(clusters, 1):
                for room_idx in cluster:
                    mark_col[room_idx] = cluster_id

            # 为每个房间计算与其他所有房间的相似度（不管是否在同一cluster）
            for i in range(group_size):
                all_similarities = []
                for j in range(group_size):
                    if i != j:
                        sim = similarity_matrix.get((i, j), 0.0)
                        all_similarities.append(f"{sim:.4f}")
                similarity_col[i] = ",".join(all_similarities)

        # 为每一行添加标记和相似度
        for i, (_, row) in enumerate(group_data.iterrows()):
            result_row = row.copy()
            result_row['标记列'] = mark_col[i]
            result_row['相似度列'] = similarity_col[i]
            results.append(result_row)

        return results

    def process_csv_file(self, input_path: str, output_path: str):
        """处理整个CSV文件"""
        logger.info(f"开始处理CSV文件: {input_path}")

        # 读取CSV文件
        df = pd.read_csv(input_path)
        logger.info(f"原始数据: {len(df)} 行")

        # 按SHotelId和GroupId分组
        grouped = df.groupby(['SHotelId', 'GroupId'])
        logger.info(f"总共分组数: {len(grouped)}")

        # 筛选分组数量>=2的数据
        filtered_groups = {name: group for name,
                           group in grouped if len(group) >= 2}
        logger.info(f"分组数量>=2的分组数: {len(filtered_groups)}")

        total_filtered_rows = sum(len(group)
                                  for group in filtered_groups.values())
        logger.info(f"筛选后的数据行数: {total_filtered_rows}")

        # 处理每个分组
        all_results = []

        for (hotel_id, group_id), group_data in tqdm(filtered_groups.items(), desc="处理分组"):
            group_size = len(group_data)
            logger.debug(
                f"处理分组 Hotel:{hotel_id}, Group:{group_id}, 大小:{group_size}")

            try:
                if group_size == 2:
                    group_results = self.process_group_of_two(group_data)
                else:
                    group_results = self.process_group_of_multiple(group_data)

                all_results.extend(group_results)

            except Exception as e:
                logger.error(
                    f"处理分组 Hotel:{hotel_id}, Group:{group_id} 时出错: {str(e)}")
                # 如果处理失败，添加原始数据但标记为错误
                for _, row in group_data.iterrows():
                    result_row = row.copy()
                    result_row['标记列'] = -1  # 错误标记
                    result_row['相似度列'] = "ERROR"
                    all_results.append(result_row)

        # 转换为DataFrame并保存
        result_df = pd.DataFrame(all_results)
        result_df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"处理完成，结果保存到: {output_path}")
        logger.info(f"输出数据: {len(result_df)} 行")

        # 输出统计信息
        self.print_statistics(result_df)

        return result_df

    def print_statistics(self, result_df):
        """打印处理统计信息"""
        logger.info("\n=== 处理统计信息 ===")

        # 标记列统计
        mark_counts = result_df['标记列'].value_counts().sort_index()
        logger.info("标记列分布:")
        for mark, count in mark_counts.items():
            if mark == -1:
                logger.info(f"  错误标记: {count}")
            else:
                logger.info(f"  标记{mark}: {count}")

        # 相似度统计
        similarity_cols = result_df['相似度列']
        numeric_similarities = []

        for sim in similarity_cols:
            if isinstance(sim, (int, float)) and not pd.isna(sim):
                numeric_similarities.append(float(sim))
            elif isinstance(sim, str) and sim.replace('.', '').replace(',', '').replace('-', '').isdigit():
                try:
                    # 尝试解析数值相似度
                    if ',' in sim:
                        # 多个相似度值，取平均
                        values = [float(x) for x in sim.split(',')]
                        numeric_similarities.append(np.mean(values))
                    else:
                        numeric_similarities.append(float(sim))
                except:
                    pass

        if numeric_similarities:
            logger.info(f"\n相似度统计:")
            logger.info(f"  数值型相似度数量: {len(numeric_similarities)}")
            logger.info(f"  平均相似度: {np.mean(numeric_similarities):.4f}")
            logger.info(f"  最大相似度: {np.max(numeric_similarities):.4f}")
            logger.info(f"  最小相似度: {np.min(numeric_similarities):.4f}")
            logger.info(
                f"  高相似度(>{SIMILARITY_THRESHOLD})数量: {sum(1 for x in numeric_similarities if x > SIMILARITY_THRESHOLD)}")


def main():
    """主函数"""
    try:
        # 初始化处理器
        processor = GroupSimilarityProcessor(MODEL_PATH)

        # 处理CSV文件
        result_df = processor.process_csv_file(INPUT_CSV_PATH, OUTPUT_CSV_PATH)

        logger.info("所有处理完成！")

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")


if __name__ == '__main__':
    main()
