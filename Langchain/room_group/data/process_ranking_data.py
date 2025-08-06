#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV数据处理脚本
功能：
1. 读取ranking_all_end.csv文件
2. 处理HIGHER数据：调换room_a和room_b，添加LOWER记录
3. 处理INCOMPARABLE数据：调换room_a和room_b，添加INCOMPARABLE记录
4. 统计数据分布和重复数量
"""

import pandas as pd
import numpy as np
from collections import Counter
import os
from tqdm import tqdm


def load_csv_data(file_path):
    """加载CSV数据"""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"成功加载数据，共 {len(df)} 行")
        return df
    except Exception as e:
        print(f"加载CSV文件失败: {e}")
        return None


def create_existing_pairs_set(df):
    """创建已存在房间对的set，用于快速查找"""
    existing_pairs = set()
    for _, row in df.iterrows():
        pair_key = (row['room_a'], row['room_b'],
                    row['s_hotel_id'], row['country_code'])
        existing_pairs.add(pair_key)
    return existing_pairs


def process_higher_data(df, existing_pairs):
    """处理HIGHER数据，添加对应的LOWER记录"""
    higher_data = df[df['comparison_result'] == 'HIGHER'].copy()
    new_records = []

    print(f"找到 {len(higher_data)} 条HIGHER记录")

    # 使用tqdm添加进度条
    for idx, row in tqdm(higher_data.iterrows(), total=len(higher_data), desc="处理HIGHER数据"):
        # 调换room_a和room_b
        new_room_a = row['room_b']
        new_room_b = row['room_a']

        # 使用set快速检查是否已存在
        pair_key = (new_room_a, new_room_b,
                    row['s_hotel_id'], row['country_code'])
        if pair_key not in existing_pairs:
            # 创建新记录
            new_record = row.copy()
            new_record['room_a'] = new_room_a
            new_record['room_b'] = new_room_b
            new_record['comparison_result'] = 'LOWER'
            new_record['reason'] = f"对应记录 {row['room_a']} vs {row['room_b']} 的对称记录"
            new_record['ratio'] = 1 - \
                row['ratio'] if pd.notna(row['ratio']) else np.nan
            new_record['is_swapped'] = True
            new_record['thinking'] = f"这是 {row['room_a']} vs {row['room_b']} 的对称记录"
            new_record['complaint_result'] = 'NO_COMPLAIN'  # 默认值

            new_records.append(new_record)
            # 更新existing_pairs以避免重复
            existing_pairs.add(pair_key)

    print(f"添加了 {len(new_records)} 条LOWER记录")
    return new_records


def process_incomparable_data(df, existing_pairs):
    """处理INCOMPARABLE数据，添加对应的INCOMPARABLE记录"""
    incomparable_data = df[df['comparison_result'] == 'INCOMPARABLE'].copy()
    new_records = []

    print(f"找到 {len(incomparable_data)} 条INCOMPARABLE记录")

    # 使用tqdm添加进度条
    for idx, row in tqdm(incomparable_data.iterrows(), total=len(incomparable_data), desc="处理INCOMPARABLE数据"):
        # 调换room_a和room_b
        new_room_a = row['room_b']
        new_room_b = row['room_a']

        # 使用set快速检查是否已存在
        pair_key = (new_room_a, new_room_b,
                    row['s_hotel_id'], row['country_code'])
        if pair_key not in existing_pairs:
            # 创建新记录
            new_record = row.copy()
            new_record['room_a'] = new_room_a
            new_record['room_b'] = new_room_b
            new_record['comparison_result'] = 'INCOMPARABLE'
            new_record['reason'] = f"对应记录 {row['room_a']} vs {row['room_b']} 的对称记录"
            new_record['ratio'] = row['ratio']  # INCOMPARABLE保持相同比例
            new_record['is_swapped'] = True
            new_record['thinking'] = f"这是 {row['room_a']} vs {row['room_b']} 的对称记录"
            # 保持相同投诉结果
            new_record['complaint_result'] = row['complaint_result']

            new_records.append(new_record)
            # 更新existing_pairs以避免重复
            existing_pairs.add(pair_key)

    print(f"添加了 {len(new_records)} 条INCOMPARABLE记录")
    return new_records


def analyze_data_distribution(df):
    """分析数据分布"""
    print("\n=== 数据分布统计 ===")

    # comparison_result分布
    result_counts = df['comparison_result'].value_counts()
    print("comparison_result分布:")
    for result, count in result_counts.items():
        print(f"  {result}: {count} ({count/len(df)*100:.2f}%)")

    # complaint_result分布
    complaint_counts = df['complaint_result'].value_counts()
    print("\ncomplaint_result分布:")
    for result, count in complaint_counts.items():
        print(f"  {result}: {count} ({count/len(df)*100:.2f}%)")

    # is_swapped分布
    swapped_counts = df['is_swapped'].value_counts()
    print("\nis_swapped分布:")
    for result, count in swapped_counts.items():
        print(f"  {result}: {count} ({count/len(df)*100:.2f}%)")


def analyze_room_duplicates(df):
    """分析room_a和room_b的重复情况"""
    print("\n=== 房间重复统计 ===")

    # 统计room_a出现次数
    room_a_counts = df['room_a'].value_counts()
    print(f"room_a唯一值数量: {room_a_counts.nunique()}")
    print(f"room_a总出现次数: {len(df)}")

    # 统计room_b出现次数
    room_b_counts = df['room_b'].value_counts()
    print(f"room_b唯一值数量: {room_b_counts.nunique()}")
    print(f"room_b总出现次数: {len(df)}")

    # 统计重复最多的房间
    print("\nroom_a出现次数最多的前10个:")
    for room, count in room_a_counts.head(10).items():
        print(f"  {room}: {count}次")

    print("\nroom_b出现次数最多的前10个:")
    for room, count in room_b_counts.head(10).items():
        print(f"  {room}: {count}次")

    # 统计同时出现在room_a和room_b中的房间
    room_a_set = set(df['room_a'].unique())
    room_b_set = set(df['room_b'].unique())
    common_rooms = room_a_set & room_b_set
    print(f"\n同时出现在room_a和room_b中的房间数量: {len(common_rooms)}")


def main():
    """主函数"""
    # 文件路径
    csv_file = "/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/ranking_all_end.csv"
    output_file = "/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/ranking_all_end_processed.csv"

    print("开始处理CSV数据...")

    # 加载数据
    df = load_csv_data(csv_file)
    if df is None:
        return

    # 显示原始数据统计
    print(f"\n原始数据统计:")
    print(f"总行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")

    # 创建已存在房间对的set
    print("创建已存在房间对的索引...")
    existing_pairs = create_existing_pairs_set(df)
    print(f"已创建 {len(existing_pairs)} 个房间对的索引")

    # 处理HIGHER数据
    higher_records = process_higher_data(df, existing_pairs)

    # 处理INCOMPARABLE数据
    incomparable_records = process_incomparable_data(df, existing_pairs)

    # 合并新记录
    all_new_records = higher_records + incomparable_records
    if all_new_records:
        new_df = pd.concat(
            [df, pd.DataFrame(all_new_records)], ignore_index=True)
        print(f"\n添加了 {len(all_new_records)} 条新记录")
        print(f"处理后总行数: {len(new_df)}")
    else:
        new_df = df
        print("\n没有添加新记录")

    # 保存处理后的数据
    print("保存处理后的数据...")
    new_df.to_csv(output_file, index=False)
    print(f"处理后的数据已保存到: {output_file}")

    # 分析数据分布
    analyze_data_distribution(new_df)

    # 分析房间重复情况
    analyze_room_duplicates(new_df)

    print("\n数据处理完成!")


if __name__ == "__main__":
    main()
