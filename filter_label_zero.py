#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取CSV文件中所有label为0的数据
"""

import pandas as pd
import os


def get_label_zero_data(csv_file_path):
    """
    从CSV文件中获取所有label为0的数据

    Args:
        csv_file_path (str): CSV文件路径

    Returns:
        pandas.DataFrame: label为0的数据
    """

    try:
        # 读取CSV文件
        print(f"正在读取文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path)

        # 显示数据基本信息
        print(f"总数据行数: {len(df)}")
        print(f"列名: {list(df.columns)}")

        # 过滤出label为0的数据
        label_zero_data = df[df['label'] == 0]

        # 显示过滤结果
        print(f"label为0的数据行数: {len(label_zero_data)}")

        return label_zero_data

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


def save_filtered_data(data, output_file):
    """
    保存过滤后的数据到新文件

    Args:
        data (pandas.DataFrame): 要保存的数据
        output_file (str): 输出文件路径
    """
    try:
        data.to_csv(output_file, index=False, encoding='utf-8')
        print(f"已保存过滤后的数据到: {output_file}")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")


def main():
    # 输入文件路径
    input_file = "/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group/room_group_train_fix_clean_filtered.csv"

    # 输出文件路径
    output_file = "/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group/label_zero_data.csv"

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 - {input_file}")
        return

    # 获取label为0的数据
    label_zero_data = get_label_zero_data(input_file)

    if label_zero_data is not None and not label_zero_data.empty:
        # 显示前几行数据
        print("\n前5行数据预览:")
        print(label_zero_data.head())

        # 保存到新文件
        save_filtered_data(label_zero_data, output_file)

        # 显示数据统计信息
        print(f"\n数据统计:")
        print(f"总行数: {len(label_zero_data)}")

        # 如果你只想在内存中使用数据而不保存文件，可以返回数据
        return label_zero_data

    else:
        print("没有找到label为0的数据或读取失败")
        return None


if __name__ == "__main__":
    filtered_data = main()
