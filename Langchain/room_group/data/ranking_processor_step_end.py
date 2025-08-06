#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房间对比数据处理脚本 - 优化版本
根据用户需求处理CSV数据，包括complaint_result和comparison_result的逻辑修改
使用set和dict优化查询性能，添加进度条显示处理进度
"""

import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import time
from functools import wraps

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 文件路径配置
BASE_PATH = Path(
    '/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data')
MAIN_FILE = BASE_PATH / 'room_review_results.csv'
RANKING_FILES = [
    BASE_PATH / 'room_ranking_results_spl_room.csv',
    BASE_PATH / 'room_ranking_results_spl_room2.csv',
    BASE_PATH / 'room_ranking_results_standard_room.csv'
]
PROCESSED_ROOM_FILE = BASE_PATH / 'processed_room_data.csv'
OUTPUT_FILE = BASE_PATH / 'ranking_all.csv'


def timer_decorator(func):
    """性能计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} 执行时间: {end_time - start_time:.2f}秒")
        return result
    return wrapper


@timer_decorator
def load_and_process_main_data():
    """加载并处理主要数据文件"""
    logger.info("正在加载主要数据文件...")

    # 使用chunks读取大文件以显示加载进度
    chunk_size = 10000
    chunks = []

    with tqdm(desc="加载CSV文件") as pbar:
        for chunk in pd.read_csv(MAIN_FILE, chunksize=chunk_size):
            chunks.append(chunk)
            pbar.update(len(chunk))

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"原始数据形状: {df.shape}")
    logger.info(f"列名: {list(df.columns)}")

    # 显示数据分布
    logger.info("原始数据分布:")
    logger.info(
        f"complaint_result分布:\n{df['complaint_result'].value_counts()}")
    logger.info(
        f"comparison_result分布:\n{df['comparison_result'].value_counts()}")

    return df


@timer_decorator
def modify_complaint_result_logic(df):
    """根据complaint_result修改comparison_result的逻辑"""
    logger.info("开始修改complaint_result相关逻辑...")

    # 保存修改前的统计
    before_stats = df['comparison_result'].value_counts()

    # 对于complaint_result为COMPLAIN的数据，如果comparison_result为HIGHER，则改为INCOMPARABLE
    mask1 = (df['complaint_result'] == 'COMPLAIN') & (
        df['comparison_result'] == 'HIGHER')
    count1 = mask1.sum()
    df.loc[mask1, 'comparison_result'] = 'INCOMPARABLE'
    logger.info(f"修改了 {count1} 条COMPLAIN+HIGHER -> COMPLAIN+INCOMPARABLE")

    # 对于complaint_result为NO_COMPLAIN的数据，如果comparison_result为INCOMPARABLE，则改为HIGHER
    mask2 = (df['complaint_result'] == 'NO_COMPLAIN') & (
        df['comparison_result'] == 'INCOMPARABLE')
    count2 = mask2.sum()
    df.loc[mask2, 'comparison_result'] = 'HIGHER'
    logger.info(
        f"修改了 {count2} 条NO_COMPLAIN+INCOMPARABLE -> NO_COMPLAIN+HIGHER")

    # 显示修改后的统计
    after_stats = df['comparison_result'].value_counts()
    logger.info("修改后comparison_result分布:")
    logger.info(f"{after_stats}")

    return df


@timer_decorator
def process_higher_data_duplication(df):
    """处理HIGHER数据的复制和调换逻辑 - 优化版本"""
    logger.info("开始处理HIGHER数据的复制和调换...")

    # 筛选出comparison_result为HIGHER的数据
    higher_data = df[df['comparison_result'] == 'HIGHER'].copy()
    logger.info(f"找到 {len(higher_data)} 条HIGHER数据")

    # 创建用于快速查找的set，包含所有room_a, room_b组合
    existing_pairs = set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建查找集合"):
        pair = (row['room_a'], row['room_b'])
        existing_pairs.add(pair)

    # 批量处理HIGHER数据
    new_rows = []
    duplicate_count = 0
    new_count = 0

    for _, row in tqdm(higher_data.iterrows(), total=len(higher_data), desc="处理HIGHER数据"):
        # 创建调换后的组合
        swapped_pair = (row['room_b'], row['room_a'])

        # 使用set进行O(1)时间复杂度的查找
        if swapped_pair in existing_pairs:
            # 如果存在，设置为HIGHER
            swapped_row = row.copy()
            swapped_row['room_a'] = row['room_b']
            swapped_row['room_b'] = row['room_a']
            swapped_row['comparison_result'] = 'HIGHER'
            new_rows.append(swapped_row)
            duplicate_count += 1
        else:
            # 如果不存在，设置为LOWER
            swapped_row = row.copy()
            swapped_row['room_a'] = row['room_b']
            swapped_row['room_b'] = row['room_a']
            swapped_row['comparison_result'] = 'LOWER'
            new_rows.append(swapped_row)
            new_count += 1

    logger.info(f"找到重复数据: {duplicate_count} 条，新增数据: {new_count} 条")

    # 将新数据添加到原数据中
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        logger.info(f"添加了 {len(new_rows)} 条新数据")

    return df


@timer_decorator
def load_equal_data_and_process(df):
    """加载EQUAL数据并处理 - 优化版本"""
    logger.info("开始处理EQUAL数据...")

    # 加载三个ranking文件中的EQUAL数据
    equal_data_list = []

    for file_path in tqdm(RANKING_FILES, desc="加载ranking文件"):
        if file_path.exists():
            temp_df = pd.read_csv(file_path)
            equal_temp = temp_df[temp_df['comparison_result']
                                 == 'EQUAL'].copy()
            equal_data_list.append(equal_temp)
            logger.info(f"从 {file_path.name} 加载了 {len(equal_temp)} 条EQUAL数据")
        else:
            logger.warning(f"文件不存在: {file_path}")

    if not equal_data_list:
        logger.warning("没有找到任何EQUAL数据")
        return df

    # 合并所有EQUAL数据
    equal_data = pd.concat(equal_data_list, ignore_index=True)
    logger.info(f"总计EQUAL数据: {len(equal_data)} 条")

    # 创建用于快速查找的字典，按comparison_result分类
    ranking_dict = defaultdict(set)
    logger.info("构建ranking_all查找字典...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建查找字典"):
        pair = (row['room_a'], row['room_b'])
        ranking_dict[row['comparison_result']].add(pair)

    # 批量处理EQUAL数据
    new_rows = []
    both_direction_count = 0
    one_direction_count = 0

    for _, row in tqdm(equal_data.iterrows(), total=len(equal_data), desc="处理EQUAL数据"):
        room_a, room_b = row['room_a'], row['room_b']
        original_pair = (room_a, room_b)
        swapped_pair = (room_b, room_a)

        # 使用set进行O(1)查找
        original_exists_higher = original_pair in ranking_dict['HIGHER']
        swapped_exists_higher = swapped_pair in ranking_dict['HIGHER']

        if not original_exists_higher:
            # 原组合不存在HIGHER，添加两个方向
            new_row1 = row.copy()
            new_row1['comparison_result'] = 'HIGHER'
            new_rows.append(new_row1)

            new_row2 = row.copy()
            new_row2['room_a'] = room_b
            new_row2['room_b'] = room_a
            new_row2['comparison_result'] = 'HIGHER'
            new_rows.append(new_row2)
            both_direction_count += 1
        elif not swapped_exists_higher:
            # 原组合存在HIGHER但调换后不存在，只添加调换后的
            new_row = row.copy()
            new_row['room_a'] = room_b
            new_row['room_b'] = room_a
            new_row['comparison_result'] = 'HIGHER'
            new_rows.append(new_row)
            one_direction_count += 1

    logger.info(
        f"添加双向EQUAL数据: {both_direction_count} 对，单向数据: {one_direction_count} 条")

    # 添加新数据
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        logger.info(f"从EQUAL数据添加了 {len(new_rows)} 条新数据")

    return df


@timer_decorator
def merge_processed_room_data(df):
    """合并processed_room_data.csv中的数据 - 优化版本"""
    logger.info("开始合并processed_room_data.csv...")

    if not PROCESSED_ROOM_FILE.exists():
        logger.warning(f"文件不存在: {PROCESSED_ROOM_FILE}")
        return df

    # 分块读取大文件
    chunk_size = 10000
    chunks = []

    logger.info("加载processed_room_data.csv...")
    with tqdm(desc="加载processed_room数据") as pbar:
        for chunk in pd.read_csv(PROCESSED_ROOM_FILE, chunksize=chunk_size):
            chunks.append(chunk)
            pbar.update(len(chunk))

    processed_df = pd.concat(chunks, ignore_index=True)
    logger.info(f"加载processed_room_data.csv: {processed_df.shape}")

    # 创建用于快速查找的set，包含所有方向的组合
    existing_pairs = set()
    logger.info("构建ranking_all双向查找集合...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建双向查找集合"):
        pair1 = (row['room_a'], row['room_b'])
        pair2 = (row['room_b'], row['room_a'])
        existing_pairs.add(pair1)
        existing_pairs.add(pair2)

    # 批量处理processed_room数据
    new_rows = []

    for _, row in tqdm(processed_df.iterrows(), total=len(processed_df), desc="检查processed_room数据"):
        room_a, room_b = row['room_a'], row['room_b']
        pair = (room_a, room_b)

        # 使用set进行O(1)查找（任一方向）
        if pair not in existing_pairs:
            new_rows.append(row)

    logger.info(f"找到 {len(new_rows)} 条不重复的processed_room数据")

    # 添加新数据
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        logger.info(f"从processed_room_data添加了 {len(new_rows)} 条新数据")

    return df


def print_final_distribution(df):
    """打印最终数据分布"""
    logger.info("=== 最终数据分布 ===")
    logger.info(f"总数据量: {len(df)} 条")
    logger.info(
        f"comparison_result分布:\n{df['comparison_result'].value_counts()}")

    if 'complaint_result' in df.columns:
        logger.info(
            f"complaint_result分布:\n{df['complaint_result'].value_counts()}")

    # 检查是否有重复的room_a, room_b组合
    duplicates = df.groupby(['room_a', 'room_b']).size()
    duplicate_count = (duplicates > 1).sum()
    logger.info(f"重复的room_a, room_b组合数量: {duplicate_count}")

    if duplicate_count > 0:
        logger.info("重复组合详情:")
        logger.info(duplicates[duplicates > 1])


@timer_decorator
def main():
    """主函数"""
    try:
        start_time = time.time()
        logger.info("=== 开始数据处理流程 ===")

        # 1. 加载和处理主要数据
        logger.info("步骤1: 加载主要数据文件")
        df = load_and_process_main_data()

        # 2. 修改complaint_result相关逻辑
        logger.info("步骤2: 修改complaint_result逻辑")
        df = modify_complaint_result_logic(df)

        # 3. 处理HIGHER数据的复制和调换
        logger.info("步骤3: 处理HIGHER数据复制调换")
        df = process_higher_data_duplication(df)

        # 4. 保存中间结果
        logger.info("步骤4: 保存中间结果...")
        with tqdm(desc="保存中间结果") as pbar:
            df.to_csv(OUTPUT_FILE, index=False)
            pbar.update(1)
        logger.info(f"中间结果已保存到: {OUTPUT_FILE}")

        # 5. 处理EQUAL数据
        logger.info("步骤5: 处理EQUAL数据")
        df = load_equal_data_and_process(df)

        # 6. 合并processed_room_data
        logger.info("步骤6: 合并processed_room数据")
        df = merge_processed_room_data(df)

        # 7. 最终保存
        logger.info("步骤7: 保存最终结果...")
        with tqdm(desc="保存最终结果") as pbar:
            df.to_csv(OUTPUT_FILE, index=False)
            pbar.update(1)
        logger.info(f"最终数据已保存到: {OUTPUT_FILE}")

        # 8. 打印最终分布
        logger.info("步骤8: 生成数据统计报告")
        print_final_distribution(df)

        total_time = time.time() - start_time
        logger.info(f"=== 数据处理完成! 总耗时: {total_time:.2f}秒 ===")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        import traceback
        logger.error(f"错误详情:\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
