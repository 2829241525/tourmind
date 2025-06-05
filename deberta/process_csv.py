#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import re
import time

# 配置
INPUT_FILE = "/home/maxon/disk2/roomMatch/room_match/deberta/data/processed_merged_cleaned_filtered.csv"
OUTPUT_FILE = "/home/maxon/disk2/roomMatch/room_match/deberta/data/processed_cleaned_spaces.csv"
COLUMNS_TO_DROP = ["hard_sample_type", "api_score"]
TEXT_COLUMNS = ["spl_room_text", "s_room_text"]


def process_csv():
    """
    处理CSV文件：
    1. 删除hard_sample_type和api_score列
    2. 将spl_room_text和s_room_text列中的多个空格替换为单个空格
    """
    start_time = time.time()

    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误：输入文件 {INPUT_FILE} 不存在！")
        return

    print(f"开始处理文件: {INPUT_FILE}")
    print(f"文件大小: {os.path.getsize(INPUT_FILE) / (1024*1024):.2f} MB")

    # 读取CSV文件
    try:
        df = pd.read_csv(INPUT_FILE)
        original_rows = len(df)
        original_cols = len(df.columns)
        print(f"原始数据: {original_rows} 行, {original_cols} 列")
    except Exception as e:
        print(f"读取CSV文件时出错: {str(e)}")
        return

    # 删除指定列（如果存在）
    for col in COLUMNS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"已删除列: {col}")
        else:
            print(f"列 {col} 不存在于数据中")

    # 处理文本列中的多个空格
    for col in TEXT_COLUMNS:
        if col in df.columns:
            print(f"处理列 {col} 中的多个空格...")
            # 使用正则表达式将多个空格替换为单个空格
            df[col] = df[col].astype(str).apply(
                lambda x: re.sub(r'\s+', ' ', x))
            print(f"列 {col} 处理完成")
        else:
            print(f"列 {col} 不存在于数据中")

    # 保存处理后的CSV文件
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"处理后数据: {len(df)} 行, {len(df.columns)} 列")
        print(f"处理后的文件已保存至: {OUTPUT_FILE}")
    except Exception as e:
        print(f"保存CSV文件时出错: {str(e)}")
        return

    end_time = time.time()
    print(f"处理完成，耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    process_csv()
