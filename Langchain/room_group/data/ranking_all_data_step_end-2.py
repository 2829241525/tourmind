import pandas as pd
import numpy as np
from collections import Counter
import re


def calculate_ratio(room_a, room_b):
    """计算room_a和room_b的共同词数占整体词数的比值"""
    # 将文本转换为小写并分词（去除标点符号）
    words_a = set(re.findall(r'\b\w+\b', str(room_a).lower()))
    words_b = set(re.findall(r'\b\w+\b', str(room_b).lower()))

    # 计算共同词数
    common_words = len(words_a.intersection(words_b))

    # 计算总词数
    total_words = len(words_a.union(words_b))

    # 计算比值
    if total_words == 0:
        return 0.0
    else:
        return common_words / total_words


def process_room_data():
    # 文件路径
    file_paths = [
        '/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/room_ranking_results_spl_room.csv',
        '/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/room_ranking_results_spl_room2.csv',
        '/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/room_ranking_results_standard_room.csv'
    ]

    # 读取并合并所有CSV文件
    print("正在读取CSV文件...")
    dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            print(f"成功读取文件: {file_path}, 数据量: {len(df)}")
            dfs.append(df)
        except Exception as e:
            print(f"读取文件失败: {file_path}, 错误: {e}")
            continue

    if not dfs:
        print("没有成功读取任何文件")
        return

    # 合并数据
    print("合并数据...")
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"合并后数据量: {len(merged_df)}")

    # 查看数据结构
    print("\n数据结构:")
    print(merged_df.head())
    print("\n列名:")
    print(merged_df.columns.tolist())
    print("\ncomparison_result的值分布:")
    print(merged_df['comparison_result'].value_counts())

    # 根据room_a, room_b去重
    print("根据room_a, room_b去重...")
    before_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['room_a', 'room_b'])
    after_dedup = len(merged_df)
    print(f"去重前: {before_dedup}, 去重后: {after_dedup}")

    # 处理数据
    processed_rows = []

    print("处理comparison_result...")
    for _, row in merged_df.iterrows():
        if row['comparison_result'] == 'EQUAL':
            # 复制为两份
            # 第一份：保持原有，comparison_result改为HIGHER
            row1 = row.copy()
            row1['comparison_result'] = 'HIGHER'
            processed_rows.append(row1)

            # 第二份：交换room_a和room_b，comparison_result设置为HIGHER
            row2 = row.copy()
            row2['room_a'] = row['room_b']
            row2['room_b'] = row['room_a']
            row2['comparison_result'] = 'HIGHER'
            processed_rows.append(row2)

        elif row['comparison_result'] == 'LOWER':
            # 交换room_a和room_b，comparison_result设置为HIGHER
            row_modified = row.copy()
            row_modified['room_a'] = row['room_b']
            row_modified['room_b'] = row['room_a']
            row_modified['comparison_result'] = 'HIGHER'
            processed_rows.append(row_modified)

        elif row['comparison_result'] in ['HIGHER', 'INCOMPARABLE']:
            # 保持原样
            processed_rows.append(row)
        else:
            print(f"未知的comparison_result值: {row['comparison_result']}")

    # 创建新的DataFrame
    final_df = pd.DataFrame(processed_rows)
    print(f"处理后数据量: {len(final_df)}")
    print("\n处理后comparison_result的值分布:")
    print(final_df['comparison_result'].value_counts())

    # 计算ratio列
    print("计算ratio列...")
    final_df['ratio'] = final_df.apply(
        lambda row: calculate_ratio(row['room_a'], row['room_b']), axis=1)

    # 保存新的CSV文件
    output_file = '/home/maxon/disk2/roomMatch/room_match/processed_room_data.csv'
    print(f"保存处理后的数据到: {output_file}")
    final_df.to_csv(output_file, index=False)

    # 显示统计信息
    print("\n最终统计信息:")
    print(f"总数据量: {len(final_df)}")
    print(
        f"HIGHER数量: {len(final_df[final_df['comparison_result'] == 'HIGHER'])}")
    print(
        f"INCOMPARABLE数量: {len(final_df[final_df['comparison_result'] == 'INCOMPARABLE'])}")
    print(
        f"ratio统计: 最小值={final_df['ratio'].min():.4f}, 最大值={final_df['ratio'].max():.4f}, 平均值={final_df['ratio'].mean():.4f}")

    print("\n前5行数据预览:")
    print(final_df.head())

    return final_df


if __name__ == "__main__":
    result_df = process_room_data()
