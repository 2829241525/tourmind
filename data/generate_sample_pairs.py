import pandas as pd
import os
import logging
from pathlib import Path
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置路径
DATA_DIR = Path('/home/maxon/disk2/roomMatch/room_match/data')
INPUT_FILE = DATA_DIR / 'grouped_room_by_qwenmax_0520_updated.csv'
OUTPUT_FILE = DATA_DIR / 'sample_pairs.csv'


def combine_text(row):
    """将RoomName和BedTypeDesc组合成文本，如果任一为空则不进行拼接"""
    room_name = str(row['RoomName']).strip()
    bed_type = str(row['BedTypeDesc']).strip()

    if pd.isna(room_name) or room_name == '':
        return bed_type if not pd.isna(bed_type) and bed_type != '' else ''
    if pd.isna(bed_type) or bed_type == '':
        return room_name

    return f"{room_name},{bed_type}"


def generate_sample_pairs(df, max_groups=10000000):
    """生成正负样本对"""
    sample_pairs = []
    processed_groups = 0
    hotel_groups = {}  # 用于存储每个酒店的所有组

    # 首先按SHotelId和InitialGroupId分组，并存储每个酒店的组信息
    for (hotel_id, group_id), group_df in df.groupby(['SHotelId', 'InitialGroupId']):
        if hotel_id not in hotel_groups:
            hotel_groups[hotel_id] = {}
        hotel_groups[hotel_id][group_id] = group_df

    # 处理每个酒店的组
    for hotel_id, groups in hotel_groups.items():
        if processed_groups >= max_groups:
            break

        # 获取当前酒店的所有zero_samples
        all_zero_samples = []
        for group_df in groups.values():
            zero_samples = group_df[(group_df['groupdf'] == 0) & (
                group_df['modeldf'] == 0)]
            if len(zero_samples) > 0:
                all_zero_samples.extend(zero_samples.to_dict('records'))

        # 生成正样本对（同组内）
        for group_id, group_df in groups.items():
            zero_samples = group_df[(group_df['groupdf'] == 0) & (
                group_df['modeldf'] == 0)]
            if len(zero_samples) >= 2:
                for i in range(len(zero_samples)):
                    for j in range(i + 1, len(zero_samples)):
                        sample_pairs.append({
                            'SHotelId': hotel_id,
                            'text1': combine_text(zero_samples.iloc[i]),
                            'text2': combine_text(zero_samples.iloc[j]),
                            'label': 1
                        })

        # 生成负样本对（同组内）
        for group_id, group_df in groups.items():
            zero_samples = group_df[(group_df['groupdf'] == 0) & (
                group_df['modeldf'] == 0)]
            if len(zero_samples) > 0:
                # 与modeldf=1的样本生成负样本
                neg_samples = group_df[(group_df['groupdf'] == 0) & (
                    group_df['modeldf'] == 1)]
                for _, zero_row in zero_samples.iterrows():
                    for _, neg_row in neg_samples.iterrows():
                        sample_pairs.append({
                            'SHotelId': hotel_id,
                            'text1': combine_text(zero_row),
                            'text2': combine_text(neg_row),
                            'label': 0
                        })

                # 与groupdf=1的样本生成负样本
                group_samples = group_df[group_df['groupdf'] == 1]
                for _, zero_row in zero_samples.iterrows():
                    for _, group_row in group_samples.iterrows():
                        sample_pairs.append({
                            'SHotelId': hotel_id,
                            'text1': combine_text(zero_row),
                            'text2': combine_text(group_row),
                            'label': 0
                        })

        # 跨InitialGroupId生成负样本
        if len(all_zero_samples) > 0:
            for i in range(len(all_zero_samples)):
                for j in range(i + 1, len(all_zero_samples)):
                    sample_pairs.append({
                        'SHotelId': hotel_id,
                        'text1': combine_text(all_zero_samples[i]),
                        'text2': combine_text(all_zero_samples[j]),
                        'label': 0
                    })

        processed_groups += 1
        if processed_groups % 100 == 0:
            logging.info(f"已处理 {processed_groups} 组数据")

    return pd.DataFrame(sample_pairs)


def main():
    try:
        # 读取CSV文件
        logging.info(f"开始读取文件: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        logging.info(f"原始数据量: {len(df)}")

        # 生成样本对
        sample_pairs_df = generate_sample_pairs(df)

        # 调整正负样本比例
        positive_samples = sample_pairs_df[sample_pairs_df['label'] == 1]
        negative_samples = sample_pairs_df[sample_pairs_df['label'] == 0]

        # 随机选择负样本，保持1:4的比例
        n_positive = len(positive_samples)
        n_negative = min(len(negative_samples), n_positive * 4)
        negative_samples = negative_samples.sample(
            n=n_negative, random_state=42)

        # 合并正负样本
        final_df = pd.concat([positive_samples, negative_samples])

        # 保存结果
        final_df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"样本对已保存至: {OUTPUT_FILE}")
        logging.info(f"正样本数量: {len(positive_samples)}")
        logging.info(f"负样本数量: {len(negative_samples)}")

    except Exception as e:
        logging.error(f"处理过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()
