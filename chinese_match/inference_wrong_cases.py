# -*- coding: utf-8 -*-
"""
mDeBERTa模型推理脚本 - 处理wrong_cases.csv
读取wrong_cases.csv文件，进行推理并输出包含相似度的新CSV文件
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

# ===================== 配置区域 =====================
# 路径配置
BASE_DIR = "/home/maxon/disk2/roomMatch/room_match/chinese_match"
MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints_mdeberta_side_fix/best_model')
INPUT_FILE = os.path.join(BASE_DIR, 'data/wrong_cases.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data/wrong_cases_with_similarity.csv')

# 模型配置
MAX_LENGTH = 128
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256
USE_FP16 = False

# 日志配置
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ===================== 日志设置 =====================
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)


class MDeBERTaInference:
    def __init__(self, model_path: str):
        """初始化推理器"""
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

                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                logger.info(f"成功从本地路径加载模型: {model_path}")

                # 记录模型的数据类型信息
                for name, param in self.model.named_parameters():
                    if param.dtype == torch.float16:
                        logger.info(f"检测到半精度参数: {name} (dtype={param.dtype})")
                        if USE_FP16 is False:
                            logger.warning(f"警告：尽管设置了使用全精度，但仍检测到半精度参数: {name}")
                        break
                else:
                    logger.info(
                        f"模型参数数据类型: {next(self.model.parameters()).dtype}")

            else:
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise

        self.model.eval()

        # 启用梯度检查点以节省显存
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("已启用梯度检查点以节省显存")

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

                # 检查并处理NaN值
                if torch.isnan(outputs.logits).any():
                    # 将NaN值替换为一个安全的负值（表示低概率）
                    outputs.logits = torch.nan_to_num(
                        outputs.logits, nan=-10.0)
                    logger.warning("检测到NaN值，已替换为安全值")

                # 获取正类的logits得分
                batch_logits = outputs.logits[:, 1]
                # 确保返回的是float32类型
                if batch_logits.dtype != torch.float32:
                    batch_logits = batch_logits.float()
                logits_list.append(batch_logits.cpu())

        return torch.cat(logits_list, dim=0)

    def calculate_similarity(self, source_embeddings, normalize_to_probability=True):
        """计算相似度，将logits转换为概率值

        Args:
            source_embeddings: 源文本嵌入或logits
            normalize_to_probability: 是否将结果归一化为0-1之间的概率值
        """
        # 获取logits作为相似度分数
        similarities = source_embeddings

        # 使用sigmoid函数将logits转换为0-1之间的概率值
        if normalize_to_probability:
            # 确保张量为float32类型
            if similarities.dtype != torch.float32:
                similarities = similarities.float()
            similarities = torch.sigmoid(similarities)

        return similarities

    def process_csv(self, input_file, output_file, batch_size=BATCH_SIZE):
        """处理CSV文件，计算相似度并输出新文件"""
        # 读取输入文件
        logger.info(f"读取输入文件: {input_file}")
        data = pd.read_csv(input_file)
        logger.info(f"读取到 {len(data)} 条数据")

        # 检查必要的列是否存在
        required_columns = ['spl_room_text', 's_room_text']
        missing_columns = [
            col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")

        # 获取文本数据
        spl_room_texts = data['spl_room_text'].tolist()
        s_room_texts = data['s_room_text'].tolist()

        # 批量处理数据
        all_similarities = []
        error_count = 0
        success_count = 0

        logger.info("开始计算文本相似度...")
        for i in tqdm(range(0, len(spl_room_texts), batch_size), desc="计算文本相似度"):
            batch_spl_texts = spl_room_texts[i:i + batch_size]
            batch_s_texts = s_room_texts[i:i + batch_size]

            try:
                # 对文本对进行编码并计算相似度
                logits = self.encode_text_pair(batch_spl_texts, batch_s_texts)
                similarities = self.calculate_similarity(
                    logits, normalize_to_probability=True)

                all_similarities.extend(similarities.cpu().numpy())
                success_count += len(batch_spl_texts)

            except Exception as e:
                error_count += len(batch_spl_texts)
                logger.error(f"处理批次 {i} 时出错: {str(e)}")
                # 为错误的数据添加默认相似度值
                all_similarities.extend([0.0] * len(batch_spl_texts))
                continue

        logger.info(f"处理完成: 成功 {success_count} 条, 失败 {error_count} 条")

        # 将相似度添加到原始数据中
        data['similarity'] = all_similarities

        # 保存结果
        logger.info(f"保存结果到: {output_file}")
        data.to_csv(output_file, index=False)
        logger.info(f"成功保存 {len(data)} 条数据到 {output_file}")

        # 输出相似度统计信息
        similarities = np.array(all_similarities)
        logger.info("\n相似度统计信息:")
        logger.info(f"最小值: {np.min(similarities):.4f}")
        logger.info(f"最大值: {np.max(similarities):.4f}")
        logger.info(f"平均值: {np.mean(similarities):.4f}")
        logger.info(f"中位数: {np.median(similarities):.4f}")
        logger.info(f"标准差: {np.std(similarities):.4f}")

        return data


def main():
    """主函数"""
    logger.info("开始推理任务...")

    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        logger.error(f"输入文件不存在: {INPUT_FILE}")
        return

    # 初始化推理器
    logger.info("初始化推理器...")
    inferencer = MDeBERTaInference(MODEL_PATH)

    # 处理CSV文件
    logger.info("开始处理CSV文件...")
    result_data = inferencer.process_csv(INPUT_FILE, OUTPUT_FILE)

    logger.info("推理任务完成！")
    logger.info(f"输出文件: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
