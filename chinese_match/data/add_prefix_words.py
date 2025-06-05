#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import os
from pathlib import Path

# 配置参数
CONFIG = {
    'data_dir': '/home/maxon/disk2/roomMatch/room_match/chinese_match/data',
    'processed_spaces_file': 'processed_cleaned_spaces.csv',
    'similarity_words_file': 'high_similarity_words.csv',
    'output_file': 'processed_cleaned_spaces_with_prefix.csv',
    'prefix_ratio': 0.08,  # 8%的数据添加前缀
    'random_seed': 42
}


def load_data():
    """加载数据文件"""
    print(f"正在加载数据文件...")

    # 读取主数据文件
    spaces_df = pd.read_csv(os.path.join(
        CONFIG['data_dir'], CONFIG['processed_spaces_file']))
    print(f"已加载 processed_cleaned_spaces 数据: {len(spaces_df)} 行")

    # 读取相似词文件
    words_df = pd.read_csv(os.path.join(
        CONFIG['data_dir'], CONFIG['similarity_words_file']))
    print(f"已加载 high_similarity_words 数据: {len(words_df)} 行")

    return spaces_df, words_df


def create_prefix_data(spaces_df, words_df):
    """创建添加前缀词的数据"""
    print(f"开始创建前缀数据...")

    # 设置随机种子
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    # 计算需要添加前缀的数据量
    total_rows = len(spaces_df)
    prefix_count = int(total_rows * CONFIG['prefix_ratio'])

    print(f"总数据量: {total_rows}")
    print(f"需要添加前缀的数据量: {prefix_count} ({CONFIG['prefix_ratio']*100}%)")

    # 随机选择需要添加前缀的行索引
    selected_indices = random.sample(range(total_rows), prefix_count)

    # 获取可用的前缀词列表
    prefix_words = words_df['word'].tolist()
    print(f"可用前缀词数量: {len(prefix_words)}")

    # 创建新的数据行
    new_rows = []

    for idx in selected_indices:
        original_row = spaces_df.iloc[idx].copy()

        # 随机选择一个前缀词并转换为小写
        prefix_word = random.choice(prefix_words).lower()

        # 创建新的文本
        new_spl_room_text = f"{prefix_word} {original_row['spl_room_text']}"
        new_s_room_text = f"{prefix_word} {original_row['s_room_text']}"

        # 创建新行
        new_row = original_row.copy()
        new_row['spl_room_text'] = new_spl_room_text
        new_row['s_room_text'] = new_s_room_text

        new_rows.append(new_row)

        if len(new_rows) % 1000 == 0:
            print(f"已处理 {len(new_rows)} / {prefix_count} 行")

    # 转换为DataFrame
    new_data_df = pd.DataFrame(new_rows)
    print(f"成功创建 {len(new_data_df)} 行新数据")

    return new_data_df


def merge_and_save(original_df, new_data_df):
    """合并原始数据和新数据，并保存"""
    print(f"正在合并数据...")

    # 合并数据（新数据添加到末尾）
    merged_df = pd.concat([original_df, new_data_df], ignore_index=True)

    print(f"合并后总数据量: {len(merged_df)} 行")
    print(f"原始数据: {len(original_df)} 行")
    print(f"新增数据: {len(new_data_df)} 行")

    # 保存到文件
    output_path = os.path.join(CONFIG['data_dir'], CONFIG['output_file'])
    merged_df.to_csv(output_path, index=False)
    print(f"数据已保存到: {output_path}")

    return merged_df


def show_examples(new_data_df, original_df):
    """显示一些示例"""
    print("\n" + "="*50)
    print("示例数据:")
    print("="*50)

    # 显示前5个例子
    for i in range(min(5, len(new_data_df))):
        new_row = new_data_df.iloc[i]
        print(f"\n示例 {i+1}:")
        print(f"新的 spl_room_text: {new_row['spl_room_text']}")
        print(f"新的 s_room_text: {new_row['s_room_text']}")
        print(f"标签: {new_row['label']}")


def main():
    """主函数"""
    print("开始执行添加前缀词任务...")
    print(f"配置信息: {CONFIG}")

    try:
        # 加载数据
        spaces_df, words_df = load_data()

        # 创建前缀数据
        new_data_df = create_prefix_data(spaces_df, words_df)

        # 合并并保存数据
        merged_df = merge_and_save(spaces_df, new_data_df)

        # 显示示例
        show_examples(new_data_df, spaces_df)

        print("\n任务完成!")

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
