#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
import os
import logging

# ========== 配置参数 ==========
INPUT_FILE = "/home/maxon/disk2/roomMatch/room_match/deberta/data/processed_merged_clean.csv"
OUTPUT_FILE_1 = "/home/maxon/disk2/roomMatch/room_match/deberta/data/filtered_pool_beach_data.csv"
OUTPUT_FILE_2 = "/home/maxon/disk2/roomMatch/room_match/deberta/data/remaining_data.csv"

# 关键词列表（用于匹配pool和beach相关的描述）
KEYWORDS = ["poolside", "pool access", "beachside",
            "beach access", "access pool", "access beach"]

# 日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 主要功能函数 ==========


def normalize_text(text):
    """
    规范化文本：转小写，将多个空格转为单个空格
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # 将多个空格转为单个空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def contains_keywords(text, keywords):
    """
    检查文本是否包含关键词
    支持词序变化（如pool access和access pool）
    """
    normalized_text = normalize_text(text)

    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)

        # 直接匹配
        if normalized_keyword in normalized_text:
            return True

        # 处理词序变化（如access pool <-> pool access）
        words = normalized_keyword.split()
        if len(words) == 2:
            reversed_keyword = f"{words[1]} {words[0]}"
            if reversed_keyword in normalized_text:
                return True

    return False


def filter_csv_data():
    """
    主函数：读取CSV文件并进行过滤
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(INPUT_FILE):
            logger.error(f"输入文件不存在: {INPUT_FILE}")
            return False

        logger.info(f"开始读取文件: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        logger.info(f"原始数据行数: {len(df)}")
        logger.info(f"数据列名: {list(df.columns)}")

        # 过滤label为1的数据
        if 'label' in df.columns:
            df_label_1 = df[df['label'] == 1].copy()
            logger.info(f"label为1的数据行数: {len(df_label_1)}")
        else:
            logger.warning("未找到'label'列，将处理所有数据")
            df_label_1 = df.copy()

        # 检查所有文本列，寻找包含关键词的记录
        text_columns = df_label_1.select_dtypes(include=['object']).columns
        logger.info(f"文本列: {list(text_columns)}")

        # 创建掩码来标识包含关键词的行
        keyword_mask = pd.Series(
            [False] * len(df_label_1), index=df_label_1.index)

        for col in text_columns:
            for idx, value in df_label_1[col].items():
                if contains_keywords(value, KEYWORDS):
                    keyword_mask[idx] = True
                    logger.debug(f"在列'{col}'中找到关键词，行索引: {idx}")

        # 分离数据
        pool_beach_data = df_label_1[keyword_mask].copy()
        remaining_data = df_label_1[~keyword_mask].copy()

        logger.info(f"包含pool/beach关键词的数据行数: {len(pool_beach_data)}")
        logger.info(f"剩余数据行数: {len(remaining_data)}")

        # 保存结果
        pool_beach_data.to_csv(OUTPUT_FILE_1, index=False, encoding='utf-8')
        logger.info(f"已保存包含pool/beach关键词的数据到: {OUTPUT_FILE_1}")

        remaining_data.to_csv(OUTPUT_FILE_2, index=False, encoding='utf-8')
        logger.info(f"已保存剩余数据到: {OUTPUT_FILE_2}")

        # 输出统计信息
        logger.info("=" * 50)
        logger.info("处理完成！统计信息：")
        logger.info(f"原始数据总行数: {len(df)}")
        logger.info(f"label=1的数据行数: {len(df_label_1)}")
        logger.info(f"包含pool/beach关键词的行数: {len(pool_beach_data)}")
        logger.info(f"剩余数据行数: {len(remaining_data)}")
        logger.info("=" * 50)

        return True

    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}")
        return False


if __name__ == "__main__":
    logger.info("开始执行CSV数据过滤程序")
    success = filter_csv_data()
    if success:
        logger.info("程序执行成功！")
    else:
        logger.error("程序执行失败！")
