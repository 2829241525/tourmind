#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

# 配置参数
CONFIG = {
    'results_dir': '/home/maxon/disk2/roomMatch/room_match/room_type_word_frequency/word_frequency_results',
    'files': [
        'room_name_frequency.csv',
        'room_name_cn_frequency.csv',
        'all_room_names_frequency.csv'
    ]
}


def analyze_frequency_file(file_path: str, file_name: str):
    """
    分析单个词频文件
    """
    print(f"\n=== {file_name} 分析结果 ===")

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        print(f"总词数: {len(df):,}")
        print(f"最低频率: {df['频率'].min()}")
        print(f"最高频率: {df['频率'].max():,}")
        print(f"平均频率: {df['频率'].mean():.2f}")
        print(f"中位数频率: {df['频率'].median():.2f}")
        print(f"标准差: {df['频率'].std():.2f}")

        # 显示频率分布
        print(f"\n频率分布:")
        freq_ranges = [1, 2, 5, 10, 50, 100, 500, 1000,
                       5000, 10000, 50000, 100000, 500000, 1000000]
        for i in range(len(freq_ranges)):
            if i == 0:
                count = len(df[df['频率'] == freq_ranges[i]])
                print(f"频率 = {freq_ranges[i]}: {count:,} 个词")
            else:
                count = len(df[(df['频率'] > freq_ranges[i-1])
                            & (df['频率'] <= freq_ranges[i])])
                print(
                    f"频率 {freq_ranges[i-1]+1}-{freq_ranges[i]}: {count:,} 个词")

        # 显示高频率词汇
        count_high = len(df[df['频率'] > freq_ranges[-1]])
        if count_high > 0:
            print(f"频率 > {freq_ranges[-1]}: {count_high:,} 个词")

        # 显示Top 10高频词
        print(f"\nTop 10 高频词:")
        top_10 = df.tail(10)[['词', '频率']].iloc[::-1]  # 反转顺序，最高频在前
        for idx, row in top_10.iterrows():
            print(f"  {row['词']}: {row['频率']:,}")

        # 显示一些中频词例子
        middle_idx = len(df) // 2
        print(f"\n中频词示例 (排名 {middle_idx-5} 到 {middle_idx+5}):")
        middle_words = df.iloc[middle_idx-5:middle_idx+5][['词', '频率']]
        for idx, row in middle_words.iterrows():
            print(f"  {row['词']}: {row['频率']}")

    except Exception as e:
        print(f"分析文件时出错: {e}")


def main():
    """
    主函数
    """
    print("房型词频分析摘要报告")
    print("=" * 50)

    for file_name in CONFIG['files']:
        file_path = os.path.join(CONFIG['results_dir'], file_name)
        analyze_frequency_file(file_path, file_name)

    print(f"\n分析完成！")
    print(f"所有结果文件位于: {CONFIG['results_dir']}")


if __name__ == "__main__":
    main()
