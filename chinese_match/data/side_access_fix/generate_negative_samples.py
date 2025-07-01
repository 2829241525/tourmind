#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import re
import time

# 配置
INPUT_FILE = "/home/maxon/disk2/roomMatch/room_match/chinese_match/data/side_access_fix/filtered_pool_beach_data.csv"
OUTPUT_FILE = "/home/maxon/disk2/roomMatch/room_match/chinese_match/data/side_access_fix/filtered_pool_beach_data_with_negatives.csv"
LABEL_NEGATIVE = 0  # 负样本标签


def contains_side_or_access(text):
    """
    检查文本是否包含side或access
    返回: ('side', 'access', 'both', 'none')
    """
    if pd.isna(text):
        return 'none'

    text_lower = str(text).lower()
    has_side = 'side' in text_lower
    has_access = 'access' in text_lower

    if has_side and has_access:
        return 'both'
    elif has_side:
        return 'side'
    elif has_access:
        return 'access'
    else:
        return 'none'


def replace_side_with_access(text):
    """
    将文本中的side替换为access，用空格分割
    """
    if pd.isna(text):
        return text

    text_str = str(text)
    # 处理 pool-side -> pool access (连字符连接的情况)
    text_str = re.sub(r'(\w+)-side\b', r'\1 access',
                      text_str, flags=re.IGNORECASE)
    # 处理 poolside -> pool access (直接连接的情况)
    text_str = re.sub(r'(\w+)side\b', r'\1 access',
                      text_str, flags=re.IGNORECASE)
    # 处理独立的 side -> access
    text_str = re.sub(r'\bside\b', 'access', text_str, flags=re.IGNORECASE)

    # 清理多余的空格
    text_str = re.sub(r'\s+', ' ', text_str).strip()
    return text_str


def replace_access_with_side(text):
    """
    将文本中的access替换为side，和前面词相连，不用空格分割
    """
    if pd.isna(text):
        return text

    text_str = str(text)
    # 处理 pool access -> poolside (空格连接的情况)
    text_str = re.sub(r'(\w+)\s+access\b', r'\1side',
                      text_str, flags=re.IGNORECASE)
    # 处理 pool-access -> poolside (连字符连接的情况)
    text_str = re.sub(r'(\w+)-access\b', r'\1side',
                      text_str, flags=re.IGNORECASE)
    # 处理独立的 access -> side
    text_str = re.sub(r'\baccess\b', 'side', text_str, flags=re.IGNORECASE)

    return text_str


def check_row_pattern(row):
    """
    检查整行数据的side/access模式
    返回: ('side_only', 'access_only', 'both', 'none')
    """
    spl_pattern = contains_side_or_access(row['spl_room_text'])
    s_pattern = contains_side_or_access(row['s_room_text'])

    # 合并两个文本的模式判断
    patterns = [spl_pattern, s_pattern]

    has_side = any(p in ['side', 'both'] for p in patterns)
    has_access = any(p in ['access', 'both'] for p in patterns)

    if has_side and has_access:
        return 'both'
    elif has_side:
        return 'side_only'
    elif has_access:
        return 'access_only'
    else:
        return 'none'


def generate_negative_samples():
    """
    生成负样本并保存到新文件
    """
    start_time = time.time()

    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误：输入文件 {INPUT_FILE} 不存在！")
        return

    print(f"开始处理文件: {INPUT_FILE}")

    # 读取CSV文件
    try:
        df = pd.read_csv(INPUT_FILE)
        original_rows = len(df)
        print(f"原始数据: {original_rows} 行")
    except Exception as e:
        print(f"读取CSV文件时出错: {str(e)}")
        return

    # 创建新的数据列表，用于存储原始数据和负样本
    new_rows = []
    negative_count = 0

    for idx, row in df.iterrows():
        # 添加原始行
        new_rows.append(row.copy())

        # 检查行模式
        pattern = check_row_pattern(row)

        if pattern == 'side_only':
            # 原本只包含side，构造两条负样本
            # 负样本1：修改spl_room_text中的side为access
            neg_row1 = row.copy()
            neg_row1['spl_room_text'] = replace_side_with_access(
                row['spl_room_text'])
            neg_row1['label'] = LABEL_NEGATIVE
            new_rows.append(neg_row1)

            # 负样本2：修改s_room_text中的side为access
            neg_row2 = row.copy()
            neg_row2['s_room_text'] = replace_side_with_access(
                row['s_room_text'])
            neg_row2['label'] = LABEL_NEGATIVE
            new_rows.append(neg_row2)

            negative_count += 2
            print(f"为第{idx+1}行生成了2个负样本 (side_only)")

        elif pattern == 'access_only':
            # 原本只包含access，构造两条负样本
            # 负样本1：修改spl_room_text中的access为side
            neg_row1 = row.copy()
            neg_row1['spl_room_text'] = replace_access_with_side(
                row['spl_room_text'])
            neg_row1['label'] = LABEL_NEGATIVE
            new_rows.append(neg_row1)

            # 负样本2：修改s_room_text中的access为side
            neg_row2 = row.copy()
            neg_row2['s_room_text'] = replace_access_with_side(
                row['s_room_text'])
            neg_row2['label'] = LABEL_NEGATIVE
            new_rows.append(neg_row2)

            negative_count += 2
            print(f"为第{idx+1}行生成了2个负样本 (access_only)")

    # 创建新的DataFrame
    result_df = pd.DataFrame(new_rows)

    # 保存结果
    try:
        result_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n处理完成！")
        print(f"原始数据行数: {original_rows}")
        print(f"生成负样本数: {negative_count}")
        print(f"最终数据行数: {len(result_df)}")
        print(f"结果文件已保存至: {OUTPUT_FILE}")
    except Exception as e:
        print(f"保存CSV文件时出错: {str(e)}")
        return

    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    generate_negative_samples()
