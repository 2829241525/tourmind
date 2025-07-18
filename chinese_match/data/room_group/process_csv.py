#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import random
from typing import List, Tuple

# 配置参数
INPUT_FILE = "room_group_llm_0717.csv"
OUTPUT_FILE = "room_group_processed.csv"
SAMPLE_RATE = 0.05  # 2%的采样率
MAX_INCREASE_RATE = 0.15  # 最大增长率6%


def has_bed_around_slash(text: str) -> bool:
    """检查/左右是否都有bed字符"""
    if pd.isna(text):
        return False

    # 查找所有/的位置
    slash_positions = [m.start() for m in re.finditer(r'/', text)]

    for pos in slash_positions:
        # 检查/左边是否有bed
        left_part = text[:pos]
        left_has_bed = bool(re.search(r'\bbed\b', left_part, re.IGNORECASE))

        # 检查/右边是否有bed
        right_part = text[pos+1:]
        right_has_bed = bool(re.search(r'\bbed\b', right_part, re.IGNORECASE))

        if left_has_bed and right_has_bed:
            return True

    return False


def process_text_replacement_advanced(text: str, pattern: str, replacement: str, operation_type: str) -> str:
    """高级文本替换处理"""
    if pd.isna(text):
        return text

    if pattern == "/":
        if operation_type == "random_partial":
            # 随机替换部分/
            slash_positions = [m.start() for m in re.finditer(r'/', text)]
            if not slash_positions:
                return text

            # 随机选择替换哪些/
            num_to_replace = random.randint(1, len(slash_positions))
            positions_to_replace = random.sample(
                slash_positions, num_to_replace)

            # 从后往前替换，避免位置偏移
            result = text
            for pos in sorted(positions_to_replace, reverse=True):
                result = result[:pos] + ' or ' + result[pos+1:]
            return result
        elif operation_type == "last_only":
            # 只替换最后一个/
            last_slash_pos = text.rfind('/')
            if last_slash_pos != -1:
                return text[:last_slash_pos] + ' or ' + text[last_slash_pos+1:]
            return text
        else:
            # 替换所有/
            return re.sub(r'/', ' or ', text)
    else:
        # 对于standard和room，随机选择替换模式
        if operation_type == "random_partial":
            # 找到所有匹配的位置
            matches = list(re.finditer(
                rf'\b{re.escape(pattern)}\b', text, re.IGNORECASE))
            if not matches:
                return text

            # 随机选择替换哪些匹配项
            num_to_replace = random.randint(1, len(matches))
            matches_to_replace = random.sample(matches, num_to_replace)

            # 从后往前替换
            result = text
            for match in sorted(matches_to_replace, key=lambda x: x.start(), reverse=True):
                result = result[:match.start()] + replacement + \
                    result[match.end():]
            return result.strip()
        else:
            # 替换所有匹配项
            return re.sub(rf'\b{re.escape(pattern)}\b', replacement, text, flags=re.IGNORECASE).strip()


def create_augmented_data(df_label1: pd.DataFrame) -> pd.DataFrame:
    """创建扩充数据"""
    augmented_rows = []
    original_count = len(df_label1)
    max_new_samples = int(original_count * MAX_INCREASE_RATE)

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    # 统计每种模式的数据量
    patterns = [
        ("standard", "standard", ""),
        ("room", "room", ""),
        ("/", "/", " or ")
    ]

    for pattern_name, search_pattern, replacement in patterns:
        print(f"处理包含 '{pattern_name}' 的数据...")

        # 找到包含该模式的数据
        if pattern_name == "/":
            mask = (df_label1['spl_room_text'].str.contains('/', na=False) |
                    df_label1['s_room_text'].str.contains('/', na=False))
        else:
            mask = (df_label1['spl_room_text'].str.contains(rf'\b{re.escape(search_pattern)}\b', na=False, regex=True, flags=re.IGNORECASE) |
                    df_label1['s_room_text'].str.contains(rf'\b{re.escape(search_pattern)}\b', na=False, regex=True, flags=re.IGNORECASE))

        candidate_data = df_label1[mask].copy()

        if len(candidate_data) == 0:
            print(f"  没有找到包含 '{pattern_name}' 的数据")
            continue

        print(f"  找到 {len(candidate_data)} 条包含 '{pattern_name}' 的数据")

        # 对于/的替换，优先选择/左右都有bed字符的数据
        if pattern_name == "/":
            priority_data = candidate_data[
                candidate_data['spl_room_text'].apply(has_bed_around_slash) |
                candidate_data['s_room_text'].apply(has_bed_around_slash)
            ].copy()

            if len(priority_data) > 0:
                print(f"  其中 {len(priority_data)} 条数据在/左右都有bed字符（优先处理）")
                # 优先从这些数据中采样
                priority_sample_ratio = 0.7  # 70%从优先数据中采样
                target_samples = int(len(candidate_data) * SAMPLE_RATE)
                remaining_quota = max_new_samples - len(augmented_rows)
                actual_samples = min(target_samples, remaining_quota)

                if actual_samples > 0:
                    priority_samples = min(
                        int(actual_samples * priority_sample_ratio), len(priority_data))
                    regular_samples = actual_samples - priority_samples

                    # 从优先数据中采样
                    if priority_samples > 0:
                        sampled_priority = priority_data.sample(
                            n=priority_samples, random_state=42)
                        candidate_data = pd.concat([sampled_priority,
                                                    candidate_data[~candidate_data.index.isin(priority_data.index)]])

        # 计算本轮可以采样的数量
        target_samples = int(len(candidate_data) * SAMPLE_RATE)
        remaining_quota = max_new_samples - len(augmented_rows)
        actual_samples = min(target_samples, remaining_quota)

        if actual_samples <= 0:
            print(f"  已达到最大扩充数量限制，跳过 '{pattern_name}'")
            continue

        print(f"  准备采样 {actual_samples} 条数据")

        # 随机采样
        sampled_data = candidate_data.sample(
            n=actual_samples, random_state=42).copy()

        # 处理文本替换
        for idx, row in sampled_data.iterrows():
            new_row = row.copy()

            # 随机决定处理策略
            process_strategy = random.choice([
                "spl_only",      # 只处理spl_room_text
                "s_only",        # 只处理s_room_text
                "both_same",     # 两个字段用同样方式处理
                "both_diff"      # 两个字段用不同方式处理
            ])

            # 随机决定替换类型
            if pattern_name == "/":
                operation_types = ["all", "random_partial", "last_only"]
            else:
                operation_types = ["all", "random_partial"]

            # 处理spl_room_text列
            if process_strategy in ["spl_only", "both_same", "both_diff"]:
                if pd.notna(row['spl_room_text']):
                    if pattern_name == "/":
                        if "/" in row['spl_room_text']:
                            op_type = random.choice(operation_types)
                            new_row['spl_room_text'] = process_text_replacement_advanced(
                                row['spl_room_text'], search_pattern, replacement, op_type)
                    elif re.search(rf'\b{re.escape(search_pattern)}\b', row['spl_room_text'], re.IGNORECASE):
                        op_type = random.choice(operation_types)
                        new_row['spl_room_text'] = process_text_replacement_advanced(
                            row['spl_room_text'], search_pattern, replacement, op_type)

            # 处理s_room_text列
            if process_strategy in ["s_only", "both_same", "both_diff"]:
                if pd.notna(row['s_room_text']):
                    if pattern_name == "/":
                        if "/" in row['s_room_text']:
                            if process_strategy == "both_same":
                                # 使用与spl相同的操作类型
                                op_type = random.choice(operation_types)
                            else:
                                op_type = random.choice(operation_types)
                            new_row['s_room_text'] = process_text_replacement_advanced(
                                row['s_room_text'], search_pattern, replacement, op_type)
                    elif re.search(rf'\b{re.escape(search_pattern)}\b', row['s_room_text'], re.IGNORECASE):
                        if process_strategy == "both_same":
                            op_type = random.choice(operation_types)
                        else:
                            op_type = random.choice(operation_types)
                        new_row['s_room_text'] = process_text_replacement_advanced(
                            row['s_room_text'], search_pattern, replacement, op_type)

            augmented_rows.append(new_row)

        print(f"  成功创建 {len(sampled_data)} 条扩充数据")

    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        return augmented_df
    else:
        return pd.DataFrame()


def main():
    print("开始处理CSV文件...")

    # 读取原始数据
    print(f"读取文件: {INPUT_FILE}")
    df_original = pd.read_csv(INPUT_FILE)
    print(f"原始数据量: {len(df_original)}")

    # 筛选label为1的数据
    df_label1 = df_original[df_original['label'] == 1].copy()
    print(f"label为1的数据量: {len(df_label1)}")

    # 创建扩充数据
    df_augmented = create_augmented_data(df_label1)

    if len(df_augmented) > 0:
        print(f"成功创建 {len(df_augmented)} 条扩充数据")

        # 合并原始数据和扩充数据
        df_final = pd.concat([df_original, df_augmented], ignore_index=True)
        print(f"最终数据量: {len(df_final)}")
        print(
            f"增长率: {(len(df_final) - len(df_original)) / len(df_original) * 100:.2f}%")

        # 保存结果
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"结果已保存到: {OUTPUT_FILE}")

        # 显示统计信息
        print("\n最终数据标签分布:")
        print(df_final['label'].value_counts())

    else:
        print("没有创建任何扩充数据")
        df_original.to_csv(OUTPUT_FILE, index=False)
        print(f"原始数据已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
