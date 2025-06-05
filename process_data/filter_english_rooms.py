#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import re
import os
import logging
from datetime import datetime

# 配置信息
CONFIG = {
    'input_file': '/home/maxon/disk2/roomMatch/room_match/data/positive_cn.csv',
    'output_file': '/home/maxon/disk2/roomMatch/room_match/data/positive_cn_filtered.csv',
    'log_file': '/home/maxon/disk2/roomMatch/room_match/process_data/filter_log.txt'
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_file']),
        logging.StreamHandler()
    ]
)


def is_english(text):
    """
    判断文本是否只包含英文字符
    """
    if not isinstance(text, str):
        return False
    # 去除空格和标点符号后检查是否只包含ASCII字符
    text = re.sub(r'[^\w\s]', '', text.strip())
    return bool(text) and all(ord(c) < 128 for c in text)


def filter_english_rooms():
    """
    过滤掉spl_room_name和s_room_name都是英文的数据行
    """
    start_time = datetime.now()
    logging.info(f"开始处理文件: {CONFIG['input_file']}")

    try:
        # 读取CSV文件
        df = pd.read_csv(CONFIG['input_file'])
        initial_count = len(df)
        logging.info(f"原始数据行数: {initial_count}")

        # 确保需要的列存在
        if 'spl_room_name' not in df.columns or 's_room_name' not in df.columns:
            logging.error("CSV文件中缺少必要的列: spl_room_name 或 s_room_name")
            return

        # 过滤数据
        filtered_df = df[~(df['spl_room_name'].apply(
            is_english) & df['s_room_name'].apply(is_english))]

        # 保存过滤后的数据
        filtered_df.to_csv(CONFIG['output_file'], index=False)

        # 记录处理结果
        filtered_count = len(filtered_df)
        removed_count = initial_count - filtered_count
        logging.info(f"处理完成! 过滤后数据行数: {filtered_count}")
        logging.info(
            f"移除了 {removed_count} 行数据 (spl_room_name和s_room_name都为英文)")

        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        logging.info(f"总处理时间: {elapsed_time:.2f} 秒")

    except Exception as e:
        logging.error(f"处理过程中发生错误: {str(e)}")


if __name__ == "__main__":
    filter_english_rooms()
