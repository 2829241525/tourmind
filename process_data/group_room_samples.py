#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import pandas as pd
from collections import defaultdict
import logging
from tqdm import tqdm
import numpy as np
import Levenshtein
import matplotlib.pyplot as plt
import requests
import re
import json
from typing import List, Dict, Tuple, Set

# 配置/home/maxon/disk2/roomMatch/room_match/data/room_groupings_20250513103209.csv
CONFIG = {
    "input_csv_path": "/home/maxon/disk2/roomMatch/room_match/data/room_groupings_20250513103209.csv",
    "output_csv_path": "/home/maxon/disk2/roomMatch/room_match/process_data/room_samples.csv",
    "processed_data_path": "/home/maxon/disk2/roomMatch/room_match/process_data/room_group_process.csv",  # 处理后的原始数据输出路径
    "positive_negative_ratio": 0.25,  # 正负样本比例 1:4
    "random_seed": 42,
    "chunk_size": 10000,  # 分块读取CSV文件的大小
    "log_file": "/home/maxon/disk2/roomMatch/room_match/process_data/generate_samples.log",
    "edit_distance_stats_file": "/home/maxon/disk2/roomMatch/room_match/process_data/edit_distance_stats.png",
    "bed_prediction_api": "http://10.0.0.110:8300/predict/",
    "enable_bed_prediction": True,  # 控制是否启用床型预测功能
    "enable_bed_analysis": True,    # 控制是否启用床型分析功能
    "enable_room_analysis": True,   # 控制是否启用房间名称分析功能
    "enable_sample_generation": True  # 控制是否生成正负样本
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def concat_room_bed(row):
    """将RoomName和BedTypeDesc拼接"""
    room_name = str(row["RoomName"]) if pd.notna(row["RoomName"]) else ""
    bed_type = str(row["BedTypeDesc"]) if pd.notna(row["BedTypeDesc"]) else ""
    # 返回小写形式
    result = f"{room_name},{bed_type}" if room_name and bed_type else room_name or bed_type
    return result.lower()


def calculate_edit_distance(text1, text2):
    """计算两个文本之间的编辑距离"""
    # 将文本统一转换为小写
    text1 = text1.lower()
    text2 = text2.lower()
    return Levenshtein.distance(text1, text2)


def calculate_jaccard_similarity(text1, text2):
    """计算两个文本之间的Jaccard相似度

    Jaccard相似度 = 交集大小 / 并集大小
    """
    # 将文本统一转换为小写
    text1 = text1.lower()
    text2 = text2.lower()

    # 将文本转换为字符集合
    set1 = set(text1)
    set2 = set(text2)

    # 计算交集和并集
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # 避免除以零
    return intersection / union if union > 0 else 0.0


def plot_edit_distance_stats(distances):
    """绘制编辑距离分布统计图"""
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, alpha=0.7, color='blue')
    plt.axvline(np.mean(distances), color='red', linestyle='dashed',
                linewidth=2, label=f'平均值: {np.mean(distances):.2f}')
    plt.axvline(np.median(distances), color='green', linestyle='dashed',
                linewidth=2, label=f'中位数: {np.median(distances):.2f}')
    plt.title('正样本编辑距离分布')
    plt.xlabel('编辑距离')
    plt.ylabel('频率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(CONFIG["edit_distance_stats_file"])
    logger.info(f"编辑距离统计图已保存至 {CONFIG['edit_distance_stats_file']}")


def predict_bed_type(df: pd.DataFrame) -> pd.DataFrame:
    """调用API预测BedTypeDesc为空的房间的床型描述"""
    logger.info("开始处理BedTypeDesc为空的房间数据")

    # 找出BedTypeDesc为空但RoomName不为空的行
    empty_bed_mask = df["BedTypeDesc"].isna() & df["RoomName"].notna()
    empty_bed_count = empty_bed_mask.sum()

    if empty_bed_count == 0:
        logger.info("没有需要预测床型的房间")
        return df

    logger.info(f"发现 {empty_bed_count} 个需要预测床型的房间")

    # 获取需要预测的房间名称
    room_names = df.loc[empty_bed_mask, "RoomName"].tolist()

    # 将预测任务分批处理，每批最多100个
    batch_size = 100
    predicted_results = []

    for i in range(0, len(room_names), batch_size):
        batch_rooms = room_names[i:i+batch_size]
        logger.info(f"处理第 {i//batch_size + 1} 批，包含 {len(batch_rooms)} 个房间")

        try:
            # 准备API请求数据
            payload = {"room_names": batch_rooms}
            headers = {
                'Content-Type': 'application/json',
                'Accept': '*/*',
                'Connection': 'keep-alive'
            }

            # 发送API请求
            response = requests.post(CONFIG["bed_prediction_api"],
                                     headers=headers,
                                     data=json.dumps(payload),
                                     timeout=30)

            # 检查请求是否成功
            if response.status_code == 200:
                result = response.json()
                if "predictions" in result:
                    predicted_results.extend(result["predictions"])
                else:
                    logger.warning(f"API响应中没有predictions字段: {result}")
            else:
                logger.warning(
                    f"API请求失败，状态码: {response.status_code}, 响应内容: {response.text}")

        except Exception as e:
            logger.error(f"调用床型预测API出错: {e}")

    # 创建预测结果的映射
    prediction_map = {}
    for item in predicted_results:
        room_name = item.get("room_name")
        predicted_bed = item.get("predicted_bed")
        confidence = item.get("confidence", 0)

        if room_name and predicted_bed and confidence > 0.5:  # 仅接受置信度>0.5的预测
            prediction_map[room_name] = predicted_bed

    # 更新DataFrame
    update_count = 0
    for idx in df.index[empty_bed_mask]:
        room_name = df.at[idx, "RoomName"]
        if room_name in prediction_map:
            df.at[idx, "BedTypeDesc"] = prediction_map[room_name]
            update_count += 1

    logger.info(f"成功预测并更新了 {update_count} 个房间的床型描述")
    return df


def analyze_bed_types(df: pd.DataFrame) -> pd.DataFrame:
    """分析BedTypeDesc中的床型和数量，标记同组内的不一致性"""
    logger.info("开始分析床型描述")

    # 数词映射
    word_to_number = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
        'nine': '9', 'ten': '10'
    }

    # 床型类型
    bed_types = ['king', 'single', 'twin', 'queen', 'double']

    # 床型映射 - 将相似床型映射到统一名称
    bed_type_mapping = {
        'single': 'twin_single',  # 将single映射到twin_single
        'twin': 'twin_single'     # 将twin映射到twin_single
    }

    # 预处理函数：转小写，替换数词
    def preprocess_bed_desc(text):
        if pd.isna(text):
            return ""

        # 转小写
        text = text.lower()

        # 替换数词为数字
        for word, number in word_to_number.items():
            # 确保替换完整单词，而不是子字符串
            text = re.sub(r'\b' + word + r'\b', number, text)

        return text

    # 提取床型和数量信息
    def extract_bed_info(text):
        if not text:
            return {}

        # 使用多种分隔符分割文本
        segments = re.split(r'[/,;\s]|and|or', text)
        segments = [s.strip() for s in segments if s.strip()]

        result = {}
        for segment in segments:
            # 查找床型
            for bed_type in bed_types:
                if bed_type in segment:
                    # 查找数量（通常在床型前面）
                    match = re.search(r'(\d+)\s*' + bed_type, segment)
                    count = match.group(1) if match else "1"  # 默认为1

                    # 使用映射后的床型名称
                    mapped_bed_type = bed_type_mapping.get(bed_type, bed_type)

                    # 记录床型和数量
                    if mapped_bed_type in result:
                        result[mapped_bed_type] = str(
                            int(result[mapped_bed_type]) + int(count))
                    else:
                        result[mapped_bed_type] = count

                    break

        return result

    # 预处理BedTypeDesc
    df['processed_bed_desc'] = df['BedTypeDesc'].apply(preprocess_bed_desc)

    # 按酒店ID和初始分组ID分组处理
    df['groupdf'] = 0  # 初始化groupdf字段

    # 分组处理
    hotel_groups = df.groupby(['SHotelId', 'InitialGroupId'])

    for (hotel_id, group_id), group_df in tqdm(hotel_groups, desc="分析床型组"):
        if len(group_df) <= 1:
            continue  # 跳过单条记录的组

        # 提取每一行的床型信息
        bed_infos = []
        for idx, row in group_df.iterrows():
            info = extract_bed_info(row['processed_bed_desc'])
            bed_infos.append((idx, info))

        # 统计每种床型信息的出现次数
        bed_info_count = defaultdict(int)
        for _, info in bed_infos:
            # 将字典转换为可哈希的元组表示
            info_tuple = tuple(sorted([(k, v) for k, v in info.items()]))
            bed_info_count[info_tuple] += 1

        # 找出出现次数最多的床型信息
        if bed_info_count:
            most_common_bed_info = max(
                bed_info_count.items(), key=lambda x: x[1])[0]

            # 将最常见床型信息转回字典形式
            most_common_info = dict(most_common_bed_info)

            # 标记与最常见床型信息不同的行
            for idx, info in bed_infos:
                # 检查是否与最常见床型信息一致
                is_consistent = True

                # 检查所有床型
                all_bed_types = set(info.keys()) | set(most_common_info.keys())

                for bed_type in all_bed_types:
                    count1 = info.get(bed_type, "0")  # 当前行的床型数量，空则为"0"
                    count2 = most_common_info.get(
                        bed_type, "0")  # 最常见的床型数量，空则为"0"

                    if count1 != count2:  # 数量不匹配，包括空值vs.有值的情况
                        is_consistent = False
                        break

                # 如果不一致，标记为1
                if not is_consistent:
                    df.at[idx, 'groupdf'] = 1

    logger.info(f"床型分析完成，标记了 {df['groupdf'].sum()} 条不一致记录")
    return df


def analyze_room_names(df: pd.DataFrame) -> pd.DataFrame:
    """分析RoomName，识别同组内的差异"""
    logger.info("开始分析房间名称")

    # 初始化roomdf字段
    df['roomdf'] = ""

    # 蕴含词组 - 这些词不会被显示为缺少或多余
    implied_words = {'bed', 'room', 'standard',
                     'smok', 'non', '1', '-', ',', ' superior', 'with', 'twin', 'or', }

    # 数词映射
    word_to_number = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
        'nine': '9', 'ten': '10'
    }

    # 移除括号内容的函数
    def remove_brackets(text):
        if not isinstance(text, str):
            return ""

        # 处理不配对的括号情况 - 如果有左括号但没有对应的右括号，则移除左括号到结尾的所有内容

        # 处理小括号 (...)
        while '(' in text:
            start = text.find('(')
            end = text.find(')', start)
            if end == -1:  # 没有找到对应的右括号
                text = text[:start]
                break
            else:
                text = text[:start] + text[end+1:]

        # 处理中括号 [...]
        while '[' in text:
            start = text.find('[')
            end = text.find(']', start)
            if end == -1:  # 没有找到对应的右括号
                text = text[:start]
                break
            else:
                text = text[:start] + text[end+1:]

        # 处理大括号 {...}
        while '{' in text:
            start = text.find('{')
            end = text.find('}', start)
            if end == -1:  # 没有找到对应的右括号
                text = text[:start]
                break
            else:
                text = text[:start] + text[end+1:]

        return text

    # 替换数词为数字
    def replace_numbers(text):
        if not isinstance(text, str):
            return ""

        # 替换数词为数字
        for word, number in word_to_number.items():
            # 确保替换完整单词，而不是子字符串
            text = re.sub(r'\b' + word + r'\b', number,
                          text, flags=re.IGNORECASE)

        return text

    # 在数字和字母之间添加空格
    def add_spaces_between_numbers_and_letters(text):
        if not isinstance(text, str):
            return ""

        # 数字后跟字母，没有空格时添加空格
        # \S 表示非空白字符，确保数字和字母之间没有空格
        text = re.sub(r'(\d)(\S*?)([a-zA-Z])', lambda m:
                      f"{m.group(1)} {m.group(2)}{m.group(3)}" if not m.group(2) else
                      f"{m.group(1)}{m.group(2)}{m.group(3)}", text)

        # 字母后跟数字，没有空格时添加空格
        text = re.sub(r'([a-zA-Z])(\S*?)(\d)', lambda m:
                      f"{m.group(1)} {m.group(3)}" if not m.group(2) else
                      f"{m.group(1)}{m.group(2)}{m.group(3)}", text)

        return text

    # 在特殊符号前后添加空格
    def add_spaces_around_special_chars(text):
        if not isinstance(text, str):
            return ""

        # 需要在前后添加空格的特殊符号
        # 扩展支持更多符号：斜杠、逗号、连字符、&符号、加号、竖线、冒号、分号、括号、等号、波浪号
        special_chars = r'[/,\-&+|:;()=~]'

        # 在特殊符号前添加空格（如果前面没有空格）
        text = re.sub(f'([^\s])({special_chars})', r'\1 \2', text)

        # 在特殊符号后添加空格（如果后面没有空格）
        text = re.sub(f'({special_chars})([^\s])', r'\1 \2', text)

        return text

    # 处理文本的函数：移除括号内容，替换数词为数字，在数字和字母间添加空格
    def process_text(text):
        if not isinstance(text, str):
            return ""

        # 先移除括号内容
        text = remove_brackets(text)

        # 在数字和字母之间添加空格
        text = add_spaces_between_numbers_and_letters(text)

        # 在特殊符号前后添加空格
        text = add_spaces_around_special_chars(text)

        # 替换数词为数字
        text = replace_numbers(text)

        # 返回处理后的文本
        return text

    # 按酒店ID和初始分组ID分组处理
    hotel_groups = df.groupby(['SHotelId', 'InitialGroupId'])

    for (hotel_id, group_id), group_df in tqdm(hotel_groups, desc="分析房间名称组"):
        if len(group_df) <= 1:
            continue  # 跳过单条记录的组

        # 按完全一致的房间名称再次分组
        name_groups = defaultdict(list)
        full_descriptions = {}  # 存储房间名和床型的完整描述
        processed_descs = {}    # 存储处理后的完整描述

        for idx, row in group_df.iterrows():
            room_name = row['RoomName']
            bed_type = row['BedTypeDesc'] if pd.notna(
                row['BedTypeDesc']) else ""

            if pd.notna(room_name) and isinstance(room_name, str):
                # 将房间名作为分组键
                name_groups[room_name].append(idx)

                # 拼接房间名和床型为完整描述
                full_desc = f"{room_name}, {bed_type}".strip()
                full_descriptions[idx] = full_desc

                # 处理完整描述并存储
                processed_descs[idx] = process_text(full_desc)

        # 如果只有一个组，跳过
        # if len(name_groups) <= 1:
        #     continue

        # 找出数量最多的组
        largest_group_name = max(
            name_groups.items(), key=lambda x: len(x[1]))[0]
        largest_indices = name_groups[largest_group_name]

        # 从最大组中获取第一个完整描述作为参考
        largest_ref_idx = largest_indices[0]
        largest_desc_processed = processed_descs[largest_ref_idx]
        largest_desc_words = set(largest_desc_processed.split())

        # 与最大组比较其他组，记录差异
        for name, indices in name_groups.items():
            if name != largest_group_name:
                for idx in indices:
                    # 获取当前处理后的完整描述
                    current_desc_processed = processed_descs[idx]
                    current_desc_words = set(current_desc_processed.split())

                    # 计算缺少和多余的词，排除蕴含词组
                    missing_words = {word for word in largest_desc_words - current_desc_words
                                     if not any(impl in word for impl in implied_words)}
                    extra_words = {word for word in current_desc_words - largest_desc_words
                                   if not any(impl in word for impl in implied_words)}

                    diff_text = ""
                    if missing_words:
                        diff_text += f"缺少: {', '.join(missing_words)}; "
                    if extra_words:
                        diff_text += f"多余: {', '.join(extra_words)}; "

                    # 更新roomdf字段
                    if diff_text:
                        df.at[idx, 'roomdf'] = diff_text.strip("; ")

    logger.info(f"房间名称分析完成，标记了 {(df['roomdf'] != '').sum()} 条差异记录")
    return df


def generate_samples():
    """生成正负样本"""
    logger.info(f"开始处理CSV文件: {CONFIG['input_csv_path']}")

    # 读取CSV文件
    df_list = []
    chunk_iter = pd.read_csv(
        CONFIG["input_csv_path"], chunksize=CONFIG["chunk_size"])
    for chunk in tqdm(chunk_iter, desc="读取CSV数据"):
        df_list.append(chunk)

    df = pd.concat(df_list, ignore_index=True)
    total_rows = len(df)
    logger.info(f"共读取 {total_rows} 行数据")

    # 将RoomName和BedTypeDesc转为小写
    logger.info("将RoomName和BedTypeDesc转为小写")
    df["RoomName"] = df["RoomName"].apply(
        lambda x: x.lower() if isinstance(x, str) else x)
    df["BedTypeDesc"] = df["BedTypeDesc"].apply(
        lambda x: x.lower() if isinstance(x, str) else x)

    # 对BedTypeDesc为空的房间进行床型预测
    if CONFIG["enable_bed_prediction"]:
        df = predict_bed_type(df)

    # 分析床型数据
    if CONFIG["enable_bed_analysis"]:
        df = analyze_bed_types(df)

    # 分析房间名称
    if CONFIG["enable_room_analysis"]:
        df = analyze_room_names(df)

    # 为每行创建RoomName和BedTypeDesc的拼接字段
    df["room_bed_text"] = df.apply(concat_room_bed, axis=1)

    if CONFIG["enable_sample_generation"]:
        # 生成正负样本
        samples = []
        positive_count = 0
        negative_count = 0

        # 设置随机种子
        random.seed(CONFIG["random_seed"])
        np.random.seed(CONFIG["random_seed"])

        # 按酒店ID和初始分组ID进行分组
        logger.info("按酒店ID和初始分组ID进行分组，开始生成样本")
        hotel_group_data = defaultdict(lambda: defaultdict(list))

        # 构建分组数据结构，存储行索引和相关信息 - 忽略roomdf字段
        for idx, row in df.iterrows():
            if pd.notna(row["SHotelId"]) and pd.notna(row["InitialGroupId"]) and row["room_bed_text"]:
                hotel_id = int(row["SHotelId"])
                group_id = int(row["InitialGroupId"])

                # 只记录索引、groupdf状态和room_bed_text，忽略roomdf状态
                hotel_group_data[hotel_id][group_id].append({
                    "idx": idx,
                    "groupdf": int(row["groupdf"]),
                    "text": row["room_bed_text"]
                })

        logger.info(f"数据已按 {len(hotel_group_data)} 个酒店分组")

        for hotel_id, groups in tqdm(hotel_group_data.items(), desc="生成样本"):
            # 1. 生成正样本：同组内groupdf为0的数据互相配对
            for group_id, items in groups.items():
                # 筛选出groupdf为0的数据
                valid_items = [item for item in items if item["groupdf"] == 0]

                if len(valid_items) == 0:
                    continue
                elif len(valid_items) == 1:
                    # 如果只有一条，与自身形成正样本
                    item = valid_items[0]
                    text = item["text"]

                    # 计算Jaccard相似度 (自身与自身相似度为1)
                    jaccard_sim = 1.0

                    # 添加样本到列表，编辑距离设为0
                    samples.append((text, text, 1, 0, jaccard_sim))
                    positive_count += 1
                else:
                    # 如果有多条，互相配对
                    for i in range(len(valid_items)):
                        for j in range(i + 1, len(valid_items)):
                            text1 = valid_items[i]["text"]
                            text2 = valid_items[j]["text"]

                            # 计算Jaccard相似度
                            jaccard_sim = calculate_jaccard_similarity(
                                text1, text2)

                            # 添加样本到列表，编辑距离设为0（去除编辑距离逻辑）
                            samples.append((text1, text2, 1, 0, jaccard_sim))
                            positive_count += 1

            # 2. 生成负样本：
            for group_id, items in groups.items():
                # 筛选出groupdf为0的数据
                valid_items = [item for item in items if item["groupdf"] == 0]

                if len(valid_items) == 0:
                    continue

                # 2.1 优先级1：同组内，groupdf为0的数据和groupdf为1的数据
                groupdf_1_items = [
                    item for item in items if item["groupdf"] == 1]

                for valid_item in valid_items:
                    for groupdf_1_item in groupdf_1_items:
                        text1 = valid_item["text"]
                        text2 = groupdf_1_item["text"]

                        # 计算Jaccard相似度
                        jaccard_sim = calculate_jaccard_similarity(
                            text1, text2)

                        # 添加样本到列表，编辑距离设为0
                        samples.append((text1, text2, 0, 0, jaccard_sim))
                        negative_count += 1

            # 2.3 优先级3：如果负样本不够，使用不同组之间的数据生成负样本
            if negative_count < positive_count / CONFIG["positive_negative_ratio"]:
                # 获取所有组的ID
                group_ids = list(groups.keys())

                if len(group_ids) > 1:  # 至少有两个组
                    for i in range(len(group_ids)):
                        for j in range(i + 1, len(group_ids)):
                            # 从第一个组获取groupdf为0的数据
                            group1_valid_items = [item for item in groups[group_ids[i]]
                                                  if item["groupdf"] == 0]

                            # 从第二个组获取所有数据
                            group2_items = groups[group_ids[j]]

                            # 生成跨组负样本
                            for item1 in group1_valid_items:
                                for item2 in group2_items:
                                    text1 = item1["text"]
                                    text2 = item2["text"]

                                    # 计算Jaccard相似度
                                    jaccard_sim = calculate_jaccard_similarity(
                                        text1, text2)

                                    # 添加样本到列表，编辑距离设为0
                                    samples.append(
                                        (text1, text2, 0, 0, jaccard_sim))
                                    negative_count += 1

        logger.info(f"生成正样本 {positive_count} 对，负样本 {negative_count} 对")

        # 调整正负样本比例
        target_negative_count = int(
            positive_count / CONFIG["positive_negative_ratio"])

        # 筛选负样本
        negative_samples = [s for s in samples if s[2] == 0]
        positive_samples = [s for s in samples if s[2] == 1]

        if len(negative_samples) > target_negative_count:
            logger.info(
                f"负样本过多，从 {len(negative_samples)} 减少到 {target_negative_count}")
            negative_samples = random.sample(
                negative_samples, target_negative_count)
        elif len(negative_samples) < target_negative_count:
            logger.warning(
                f"负样本不足，只有 {len(negative_samples)} 对，目标为 {target_negative_count} 对")

        # 合并正负样本
        all_samples = positive_samples + negative_samples
        random.shuffle(all_samples)

        # 创建DataFrame
        samples_df = pd.DataFrame(all_samples, columns=[
            "text1", "text2", "label", "edit_distance", "jaccard_similarity"])

        # 去除text1和text2互换的逻辑，直接保存原始样本
        logger.info(f"样本总数量: {len(samples_df)}")

        # 保存样本数据
        samples_df.to_csv(CONFIG["output_csv_path"], index=False)
        logger.info(
            f"样本生成完成，共 {len(samples_df)} 对，保存至 {CONFIG['output_csv_path']}")
    else:
        logger.info("样本生成已禁用，跳过样本生成步骤")
        positive_count = 0
        negative_count = 0

    # 调整列的顺序并保存处理后的原始数据
    if 'roomdf' in df.columns and 'groupdf' in df.columns:
        # 确保roomdf列在InitialGroupId后面，groupdf列在roomdf后面
        cols = list(df.columns)
        initial_group_idx = cols.index(
            'InitialGroupId') if 'InitialGroupId' in cols else -1

        if initial_group_idx >= 0:
            cols.remove('roomdf')
            cols.remove('groupdf')
            cols.insert(initial_group_idx + 1, 'roomdf')
            cols.insert(initial_group_idx + 2, 'groupdf')
            df = df[cols]

        # 保存处理后的原始数据
        df.to_csv(CONFIG["processed_data_path"], index=False)
        logger.info(f"处理后的原始数据已保存至 {CONFIG['processed_data_path']}")

    return positive_count, negative_count


if __name__ == "__main__":
    pos_count, neg_count = generate_samples()
    print(
        f"生成完成！正样本: {pos_count}，负样本: {neg_count}，正负比例: 1:{neg_count/pos_count if pos_count else 0:.2f}")
