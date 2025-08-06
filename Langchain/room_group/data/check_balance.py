#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查HIGHER和LOWER数据平衡性
"""

import pandas as pd
import numpy as np


def load_and_analyze_data(file_path):
    """加载并分析数据"""
    print("加载数据...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"数据总行数: {len(df)}")

    # 分析comparison_result分布
    result_counts = df['comparison_result'].value_counts()
    print("\n=== comparison_result分布 ===")
    for result, count in result_counts.items():
        print(f"{result}: {count} ({count/len(df)*100:.2f}%)")

    # 分析is_swapped分布
    swapped_counts = df['is_swapped'].value_counts()
    print("\n=== is_swapped分布 ===")
    for result, count in swapped_counts.items():
        print(f"{result}: {count} ({count/len(df)*100:.2f}%)")

    return df


def check_balance(df):
    """检查HIGHER和LOWER的平衡性"""
    print("\n=== 平衡性检查 ===")

    # 原始HIGHER数据（is_swapped=False）
    original_higher = df[(df['comparison_result'] == 'HIGHER')
                         & (df['is_swapped'] == False)]
    print(f"原始HIGHER数据: {len(original_higher)} 条")

    # 原始LOWER数据（is_swapped=False）
    original_lower = df[(df['comparison_result'] == 'LOWER')
                        & (df['is_swapped'] == False)]
    print(f"原始LOWER数据: {len(original_lower)} 条")

    # 新增的LOWER数据（is_swapped=True）
    new_lower = df[(df['comparison_result'] == 'LOWER')
                   & (df['is_swapped'] == True)]
    print(f"新增LOWER数据: {len(new_lower)} 条")

    # 总HIGHER数据
    total_higher = df[df['comparison_result'] == 'HIGHER']
    print(f"总HIGHER数据: {len(total_higher)} 条")

    # 总LOWER数据
    total_lower = df[df['comparison_result'] == 'LOWER']
    print(f"总LOWER数据: {len(total_lower)} 条")

    # 检查平衡性
    print(f"\n平衡性分析:")
    print(
        f"原始HIGHER vs 原始LOWER: {len(original_higher)} vs {len(original_lower)}")
    print(f"总HIGHER vs 总LOWER: {len(total_higher)} vs {len(total_lower)}")

    if len(total_higher) == len(total_lower):
        print("✓ HIGHER和LOWER数据完全平衡")
    else:
        print(
            f"✗ HIGHER和LOWER数据不平衡，差异: {abs(len(total_higher) - len(total_lower))}")

    return {
        'original_higher': len(original_higher),
        'original_lower': len(original_lower),
        'new_lower': len(new_lower),
        'total_higher': len(total_higher),
        'total_lower': len(total_lower)
    }


def analyze_room_pairs(df):
    """分析房间对的分布"""
    print("\n=== 房间对分析 ===")

    # 创建房间对
    df['room_pair'] = df.apply(lambda x: (
        x['room_a'], x['room_b'], x['s_hotel_id'], x['country_code']), axis=1)

    # 统计每个房间对的出现次数
    pair_counts = df['room_pair'].value_counts()

    print(f"唯一房间对数量: {len(pair_counts)}")
    print(f"平均每个房间对出现次数: {pair_counts.mean():.2f}")
    print(f"房间对出现次数分布:")
    print(f"  出现1次: {(pair_counts == 1).sum()}")
    print(f"  出现2次: {(pair_counts == 2).sum()}")
    print(f"  出现3次及以上: {(pair_counts >= 3).sum()}")

    # 检查是否有重复的房间对
    duplicate_pairs = pair_counts[pair_counts > 1]
    if len(duplicate_pairs) > 0:
        print(f"\n发现 {len(duplicate_pairs)} 个重复的房间对:")
        for pair, count in duplicate_pairs.head(10).items():
            print(f"  {pair}: {count}次")

    return pair_counts


def check_specific_issues(df):
    """检查具体问题"""
    print("\n=== 具体问题检查 ===")

    # 检查是否有HIGHER记录但没有对应的LOWER记录
    higher_pairs = set()
    for _, row in df[(df['comparison_result'] == 'HIGHER') & (df['is_swapped'] == False)].iterrows():
        pair = (row['room_a'], row['room_b'],
                row['s_hotel_id'], row['country_code'])
        higher_pairs.add(pair)

    lower_pairs = set()
    for _, row in df[df['comparison_result'] == 'LOWER'].iterrows():
        pair = (row['room_a'], row['room_b'],
                row['s_hotel_id'], row['country_code'])
        lower_pairs.add(pair)

    # 检查HIGHER对应的反向对是否在LOWER中
    missing_lower = 0
    for pair in higher_pairs:
        reverse_pair = (pair[1], pair[0], pair[2], pair[3])  # 调换room_a和room_b
        if reverse_pair not in lower_pairs:
            missing_lower += 1

    print(f"HIGHER记录数量: {len(higher_pairs)}")
    print(f"LOWER记录数量: {len(lower_pairs)}")
    print(f"缺少对应LOWER记录的HIGHER数量: {missing_lower}")

    if missing_lower > 0:
        print("⚠️ 发现一些HIGHER记录缺少对应的LOWER记录")
    else:
        print("✓ 所有HIGHER记录都有对应的LOWER记录")


def main():
    """主函数"""
    # 文件路径
    processed_file = "/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/ranking_all_end_processed.csv"

    print("开始检查数据平衡性...")

    # 加载数据
    df = load_and_analyze_data(processed_file)

    # 检查平衡性
    balance_stats = check_balance(df)

    # 分析房间对
    pair_counts = analyze_room_pairs(df)

    # 检查具体问题
    check_specific_issues(df)

    print("\n检查完成!")


if __name__ == "__main__":
    main()
