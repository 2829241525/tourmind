# -*- coding: utf-8 -*-
"""
房间分组相似度处理脚本
根据SHotelId进行分组，计算组内房间的相似度并进行标记
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
INPUT_CSV_PATH = "/home/maxon/disk2/roomMatch/room_match/chinese_match/group_eval/s_room_2025-07-21_095100.csv"
OUTPUT_CSV_PATH = "/home/maxon/disk2/roomMatch/room_match/chinese_match/group_eval/s_room_eval.csv"

# 处理配置
MAX_HOTELS = 200000  # 限制处理的酒店数量
HOTEL_BATCH_SIZE = 1000  # 每批处理的酒店数量，减小批次大小以便更频繁地显示进度

# 模型配置
MAX_LENGTH = 128
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256
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

    def get_similarity_for_pairs(self, text1_list, text2_list):
        """批量计算文本对的相似度"""
        logits = self.encode_text_pair(text1_list, text2_list)
        similarities = self.calculate_similarity(
            logits, normalize_to_probability=True)
        return similarities.numpy()

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


def process_hotel_batch(processor, hotel_batch_data, batch_idx):
    """批量处理多个酒店的房间"""
    all_results = []
    total_pairs = 0
    current_pairs = 0

    # 首先计算总文本对数量
    for hotel_id, hotel_data in hotel_batch_data:
        group_size = len(hotel_data)
        if group_size >= 2:
            total_pairs += (group_size * (group_size - 1)) // 2

    if total_pairs == 0:
        logger.info(f"批次 {batch_idx} 中没有需要处理的文本对")
        return all_results

    logger.info(f"开始处理批次 {batch_idx}，共 {total_pairs} 个文本对需要比较")
    progress_bar = tqdm(total=total_pairs, desc=f"批次 {batch_idx} 文本相似度计算")

    try:
        for hotel_id, hotel_data in hotel_batch_data:
            try:
                # 获取所有房间的文本
                room_texts = []
                for _, row in hotel_data.iterrows():
                    text = processor.combine_room_text(
                        row['room_name'], row['bed_type_desc'])
                    room_texts.append(text)

                group_size = len(room_texts)
                if group_size < 2:  # 跳过只有一个房间的酒店
                    continue

                # 准备所有可能的文本对
                text1_list = []
                text2_list = []
                pair_indices = []

                for i, j in itertools.combinations(range(group_size), 2):
                    text1_list.append(room_texts[i])
                    text2_list.append(room_texts[j])
                    pair_indices.append((i, j))

                # 批量计算相似度
                similarity_matrix = {}
                for start_idx in range(0, len(text1_list), BATCH_SIZE):
                    end_idx = min(start_idx + BATCH_SIZE, len(text1_list))
                    batch_text1 = text1_list[start_idx:end_idx]
                    batch_text2 = text2_list[start_idx:end_idx]
                    batch_pairs = pair_indices[start_idx:end_idx]

                    similarities = processor.predictor.get_similarity_for_pairs(
                        batch_text1, batch_text2)

                    for (i, j), sim in zip(batch_pairs, similarities):
                        similarity_matrix[(i, j)] = sim
                        similarity_matrix[(j, i)] = sim

                    # 更新进度条
                    batch_size = end_idx - start_idx
                    current_pairs += batch_size
                    progress_bar.update(batch_size)

                # 聚类处理
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

                        # 计算与同组其他房间的相似度
                        similarities = []
                        for other_idx in cluster:
                            if other_idx != room_idx:
                                sim = similarity_matrix.get(
                                    (room_idx, other_idx), 0.0)
                                similarities.append(f"{sim:.4f}")
                        similarity_col[room_idx] = ",".join(similarities)

                # 为每一行添加标记和相似度
                for i, (_, row) in enumerate(hotel_data.iterrows()):
                    result_row = row.copy()
                    result_row['标记列'] = mark_col[i]
                    result_row['相似度列'] = similarity_col[i]
                    all_results.append(result_row)

            except Exception as e:
                logger.error(f"处理酒店 {hotel_id} 时出错: {str(e)}")
                # 如果处理失败，添加原始数据但标记为错误
                for _, row in hotel_data.iterrows():
                    result_row = row.copy()
                    result_row['标记列'] = -1
                    result_row['相似度列'] = "ERROR"
                    all_results.append(result_row)
    finally:
        progress_bar.close()
        logger.info(
            f"批次 {batch_idx} 处理完成，共处理 {current_pairs}/{total_pairs} 个文本对")

    return all_results


def main():
    """主函数"""
    try:
        # 初始化处理器
        processor = GroupSimilarityProcessor(MODEL_PATH)

        # 读取CSV文件
        logger.info(f"读取CSV文件: {INPUT_CSV_PATH}")
        df = pd.read_csv(INPUT_CSV_PATH, on_bad_lines='skip', escapechar='\\')
        logger.info(f"原始数据: {len(df)} 行")

        # 按s_hotel_id分组
        grouped = df.groupby('s_hotel_id')
        total_hotels = len(grouped)
        logger.info(f"酒店总数: {total_hotels}，将处理其中的 {MAX_HOTELS} 家酒店")

        # 批量处理酒店
        all_results = []
        hotel_groups = list(grouped)[:MAX_HOTELS]  # 限制处理的酒店数量
        total_hotels = len(hotel_groups)

        for batch_start in range(0, total_hotels, HOTEL_BATCH_SIZE):
            batch_end = min(batch_start + HOTEL_BATCH_SIZE, total_hotels)
            batch_idx = batch_start // HOTEL_BATCH_SIZE + 1
            total_batches = (total_hotels + HOTEL_BATCH_SIZE -
                             1) // HOTEL_BATCH_SIZE

            logger.info(
                f"\n开始处理第 {batch_idx}/{total_batches} 批酒店 ({batch_end-batch_start} 家)")
            hotel_batch = hotel_groups[batch_start:batch_end]

            batch_results = process_hotel_batch(
                processor, hotel_batch, batch_idx)
            all_results.extend(batch_results)

            # 定期保存中间结果
            temp_df = pd.DataFrame(all_results)
            # 对中间结果也进行排序
            temp_df = temp_df.sort_values(['s_hotel_id', '标记列'])
            temp_df.to_csv(f"{OUTPUT_CSV_PATH}.temp",
                           index=False, encoding='utf-8')
            logger.info(f"已保存临时结果，当前已处理: {batch_end}/{total_hotels} 家酒店")

        # 保存最终结果
        result_df = pd.DataFrame(all_results)

        # 按s_hotel_id分组后，根据标记列排序
        logger.info("对结果进行排序...")
        result_df = result_df.sort_values(['s_hotel_id', '标记列'])

        # 保存排序后的结果
        result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')

        logger.info(f"处理完成，结果保存到: {OUTPUT_CSV_PATH}")
        logger.info(f"输出数据: {len(result_df)} 行")

        # 输出统计信息
        print_statistics(result_df)

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")


def print_statistics(result_df):
    """打印处理统计信息"""
    logger.info("\n=== 处理统计信息 ===")

    # 按酒店ID和标记列分组统计
    group_sizes = result_df.groupby(['s_hotel_id', '标记列']).size()
    size_distribution = group_sizes.value_counts().sort_index()
    total_groups = len(group_sizes)

    logger.info("\n分组大小统计:")
    logger.info(f"总分组数: {total_groups}")

    # 统计不同大小的分组数量
    for group_size, count in size_distribution.items():
        percentage = (count / total_groups) * 100
        logger.info(f"{group_size}个房间的分组数: {count} 个 (占比: {percentage:.2f}%)")

    # 计算平均每个酒店的分组数
    hotel_group_counts = result_df.groupby('s_hotel_id')['标记列'].nunique()
    avg_groups_per_hotel = hotel_group_counts.mean()
    max_groups_per_hotel = hotel_group_counts.max()
    min_groups_per_hotel = hotel_group_counts.min()

    logger.info("\n每个酒店的分组统计:")
    logger.info(f"平均每个酒店的分组数: {avg_groups_per_hotel:.2f}")
    logger.info(f"最多分组数: {max_groups_per_hotel}")
    logger.info(f"最少分组数: {min_groups_per_hotel}")

    # 相似度统计
    logger.info("\n相似度统计:")
    valid_similarities = []
    for sim_str in result_df['相似度列']:
        if isinstance(sim_str, str) and sim_str != "ERROR":
            sims = [float(x) for x in sim_str.split(',') if x]
            if sims:
                valid_similarities.extend(sims)

    if valid_similarities:
        logger.info(f"有效相似度数量: {len(valid_similarities)}")
        logger.info(f"平均相似度: {np.mean(valid_similarities):.4f}")
        logger.info(f"最大相似度: {np.max(valid_similarities):.4f}")
        logger.info(f"最小相似度: {np.min(valid_similarities):.4f}")
        logger.info(f"相似度中位数: {np.median(valid_similarities):.4f}")
        logger.info(
            f"高相似度(>{SIMILARITY_THRESHOLD})比例: {sum(1 for x in valid_similarities if x > SIMILARITY_THRESHOLD)/len(valid_similarities)*100:.2f}%")

    # 输出一些示例分组
    logger.info("\n分组示例:")
    # 随机选择3个不同大小的分组
    sample_sizes = [2, 5, 9]  # 选择不同大小的分组作为示例
    for size in sample_sizes:
        size_groups = group_sizes[group_sizes == size]
        if not size_groups.empty:
            sample_group = size_groups.sample(n=1)
            hotel_id, mark = sample_group.index[0]
            sample_rooms = result_df[
                (result_df['s_hotel_id'] == hotel_id) &
                (result_df['标记列'] == mark)
            ]
            logger.info(f"\n{size}个房间的分组示例 (酒店ID: {hotel_id}, 标记: {mark}):")
            for _, room in sample_rooms.iterrows():
                room_text = f"{room['room_name']} {room['bed_type_desc']}".strip(
                )
                logger.info(f"  - {room_text}")


if __name__ == '__main__':
    main()
