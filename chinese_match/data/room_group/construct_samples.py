#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from itertools import combinations
from collections import defaultdict


def construct_room_text(room_name, bed_type_desc):
    """
    构造房间文本，将RoomName和BedTypeDesc用逗号拼接
    """
    room_name = str(room_name) if pd.notna(room_name) else ""
    bed_type_desc = str(bed_type_desc) if pd.notna(bed_type_desc) else ""
    return f"{room_name},{bed_type_desc}"


def generate_positive_samples_pairwise(hotel_id, room_texts):
    """
    根据房间文本列表生成正样本（配对方式，参考process_group_size_2）
    """
    samples = []
    for i, text1 in enumerate(room_texts):
        for j, text2 in enumerate(room_texts):
            if i < j:  # 避免重复和自己配对
                # 生成两个方向的正样本
                samples.extend([
                    {
                        'SHotelId': hotel_id,
                        'spl_room_text': text1,
                        's_room_text': text2,
                        'label': 1
                    },
                    {
                        'SHotelId': hotel_id,
                        'spl_room_text': text2,
                        's_room_text': text1,
                        'label': 1
                    }
                ])
    return samples


def generate_negative_samples_pairwise(hotel_id, room_texts1, room_texts2):
    """
    根据两组房间文本列表生成负样本（配对方式，参考process_group_size_2）
    """
    samples = []
    for text1 in room_texts1:
        for text2 in room_texts2:
            # 生成两个方向的负样本
            samples.extend([
                {
                    'SHotelId': hotel_id,
                    'spl_room_text': text1,
                    's_room_text': text2,
                    'label': 0
                },
                {
                    'SHotelId': hotel_id,
                    'spl_room_text': text2,
                    's_room_text': text1,
                    'label': 0
                }
            ])
    return samples


def process_group_size_2(group_data):
    """
    处理组内数量为2的情况
    """
    samples = []
    hotel_id = group_data['SHotelId'].iloc[0]

    # 获取两条记录的newGroup值
    new_groups = group_data['newGroup'].fillna('')

    # 构造房间文本
    room_texts = []
    for _, row in group_data.iterrows():
        room_text = construct_room_text(row['RoomName'], row['BedTypeDesc'])
        room_texts.append(room_text)

    # 判断是否为空或数字相同
    if new_groups.iloc[0] == '' and new_groups.iloc[1] == '':
        # 两个都为空 → 生成负样本
        samples.extend([
            {
                'SHotelId': hotel_id,
                'spl_room_text': room_texts[0],
                's_room_text': room_texts[1],
                'label': 0
            },
            {
                'SHotelId': hotel_id,
                'spl_room_text': room_texts[1],
                's_room_text': room_texts[0],
                'label': 0
            }
        ])
    elif (new_groups.iloc[0] != '' and new_groups.iloc[1] != '' and
          str(new_groups.iloc[0]) == str(new_groups.iloc[1])):
        # 两个都不为空且相同 → 生成正样本
        samples.extend([
            {
                'SHotelId': hotel_id,
                'spl_room_text': room_texts[0],
                's_room_text': room_texts[1],
                'label': 1
            },
            {
                'SHotelId': hotel_id,
                'spl_room_text': room_texts[1],
                's_room_text': room_texts[0],
                'label': 1
            }
        ])
    else:
        # 其他情况 → 生成负样本
        samples.extend([
            {
                'SHotelId': hotel_id,
                'spl_room_text': room_texts[0],
                's_room_text': room_texts[1],
                'label': 0
            },
            {
                'SHotelId': hotel_id,
                'spl_room_text': room_texts[1],
                's_room_text': room_texts[0],
                'label': 0
            }
        ])

    return samples


def process_group_size_gt_2(group_data):
    """
    处理组内数量超过2的情况
    先进行去重，再分组。组内数量超过1的，都要生成正样本。不同组都要生成负样本
    """
    samples = []
    hotel_id = group_data['SHotelId'].iloc[0]

    # 首先收集所有房间文本和对应的newGroup信息
    room_group_pairs = []
    for _, row in group_data.iterrows():
        new_group = row['newGroup']
        room_text = construct_room_text(row['RoomName'], row['BedTypeDesc'])

        # 对所有newGroup为空值的，统一标记为99
        if pd.isna(new_group) or str(new_group).strip() == '':
            room_group_pairs.append((room_text, '99'))
        else:
            room_group_pairs.append((room_text, str(new_group)))

    # 先进行去重（基于房间文本和newGroup的组合去重）
    unique_pairs = list(set(room_group_pairs))

    # 再根据newGroup分组
    new_group_dict = defaultdict(list)
    for room_text, new_group in unique_pairs:
        new_group_dict[new_group].append(room_text)

    # 生成同组内的正样本（组内数量超过1的，都要生成正样本）
    for new_group, room_texts in new_group_dict.items():
        if len(room_texts) > 1:
            positive_samples = generate_positive_samples_pairwise(
                hotel_id, room_texts)
            samples.extend(positive_samples)

    # 生成不同组之间的负样本（所有不同组都要生成负样本）
    new_group_keys = list(new_group_dict.keys())
    for i in range(len(new_group_keys)):
        for j in range(i + 1, len(new_group_keys)):
            group1_texts = new_group_dict[new_group_keys[i]]
            group2_texts = new_group_dict[new_group_keys[j]]
            # 不同组都要生成负样本，无论数量多少
            negative_samples = generate_negative_samples_pairwise(
                hotel_id, group1_texts, group2_texts)
            samples.extend(negative_samples)

    return samples


def construct_samples(csv_file_path):
    """
    主函数：构造正负样本
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 根据SHotelId和GroupId分组
    grouped = df.groupby(['SHotelId', 'GroupId'])

    all_samples = []

    for (hotel_id, group_id), group_data in grouped:
        group_size = len(group_data)

        if group_size == 2:
            # 处理组内数量为2的情况
            samples = process_group_size_2(group_data)
            all_samples.extend(samples)

        elif group_size > 2:
            # 处理组内数量超过2的情况
            samples = process_group_size_gt_2(group_data)
            all_samples.extend(samples)

        # group_size == 1的情况暂不处理

    # 转换为DataFrame
    result_df = pd.DataFrame(all_samples)

    return result_df


def main():
    """
    主程序入口
    """
    csv_file_path = "/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group/问题房型.csv"

    # 构造样本
    samples_df = construct_samples(csv_file_path)

    # 显示统计信息
    print(f"总样本数: {len(samples_df)}")
    print(f"正样本数: {len(samples_df[samples_df['label'] == 1])}")
    print(f"负样本数: {len(samples_df[samples_df['label'] == 0])}")

    # 显示前几条样本
    print("\n前5条样本:")
    print(samples_df.head())

    # 保存结果
    output_path = "/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group/constructed_samples.csv"
    samples_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n样本已保存到: {output_path}")


if __name__ == "__main__":
    main()
