#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
床位数据增强脚本
对于spl_room_text数据中[bed]不为[unknown]的数据，有5%的概率新增一行
增加的行数据会将[bed]内的数据放到[room]数据的结尾，同时[bed]改为[unknown]
"""

import os
import logging
import pandas as pd
import numpy as np
import argparse
import re
from tqdm import tqdm
from datetime import datetime
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "logs/room_bed_augmentation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 默认配置参数/home/maxon/disk2/roomMatch/room_match/results/processed_country_with_predictions_20250331_033223.csv
CONFIG = {
    "input_file": "/home/maxon/disk2/roomMatch/room_match/results/processed_country_with_predictions_20250331_033223.csv",
    "output_dir": "/home/maxon/disk2/roomMatch/room_match/results",
    "augmentation_ratio": 0.10,  # 增强数据的比例（5%）
    "random_seed": 42,  # 随机种子，确保结果可复现
}


def load_data(input_file):
    """加载需要处理的数据"""
    logger.info(f"从 {input_file} 加载数据")

    try:
        df = pd.read_csv(input_file)
        # 检查必要的列是否存在
        required_columns = ["spl_room_text"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"输入文件中没有找到列 '{col}'")
                raise ValueError(f"输入文件中没有找到列 '{col}'")

        # 保留非空房间名
        valid_data = df[df["spl_room_text"].notna()].copy()

        logger.info(f"加载数据完成，共 {len(valid_data)} 行有效数据")
        return valid_data
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        raise


def extract_room_bed_parts(text):
    """从标记文本中提取房间名称和床位部分"""
    if "[room]" in text and "[bed]" in text:
        parts = text.split("[bed]")
        room_part = parts[0]
        bed_part = "[bed]" + parts[1] if len(parts) > 1 else "[bed] [unknown]"

        # 提取纯文本部分
        room_text = room_part.replace("[room]", "").strip()
        bed_text = bed_part.replace("[bed]", "").strip()

        if "bed" in room_part:
            return None, None, None, None
        return room_part, bed_part, room_text, bed_text
    return None, None, None, None


def augment_bed_data(input_file, output_dir, augmentation_ratio=0.05, random_seed=42):
    """处理数据并进行床位数据增强"""
    # 设置随机种子以确保结果可复现
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    data = load_data(input_file)

    # 创建一个新的DataFrame用于保存增强后的数据
    augmented_data = data.copy()

    # 记录需要增强的行
    rows_to_augment = []

    # 筛选出[bed]不为[unknown]的记录
    valid_beds = []
    for idx, row in data.iterrows():
        spl_room_text = row["spl_room_text"]
        if isinstance(spl_room_text, str) and "[bed]" in spl_room_text and "[unknown]" not in spl_room_text :
            room_part, bed_part, room_text, bed_text = extract_room_bed_parts(
                spl_room_text)
            if room_part and bed_part and room_text and bed_text:
                # 确保床位不是空的
                if bed_text.strip():
                    valid_beds.append((idx, row))

    logger.info(f"发现 {len(valid_beds)} 条床位不为[unknown]的记录")

    if len(valid_beds) == 0:
        logger.warning("没有找到需要处理的记录，处理终止")
        return None

    # 随机选择5%的数据进行增强
    augmentation_count = int(len(valid_beds) * augmentation_ratio)
    selected_indices = random.sample(
        range(len(valid_beds)), augmentation_count)
    selected_records = [valid_beds[i] for i in selected_indices]

    logger.info(
        f"将对 {augmentation_count} 条数据（{augmentation_ratio*100:.1f}%）进行增强")

    # 创建新的增强数据
    new_rows = []

    for orig_idx, orig_row in selected_records:
        spl_room_text = orig_row["spl_room_text"]
        room_part, bed_part, room_text, bed_text = extract_room_bed_parts(
            spl_room_text)

        # 创建新的房间文本，将床位信息添加到房间名称结尾
        new_room_text = f"[room] {room_text} {bed_text}"
        new_bed_text = "[bed] [unknown]"
        augmented_text = new_room_text + " " + new_bed_text

        # 创建新行，复制原行的所有内容
        new_row = orig_row.copy()
        new_row["spl_room_text"] = augmented_text
        new_row["is_augmented"] = 1  # 添加标记表示这是增强的数据

        new_rows.append(new_row)

    # 如果有新行，则添加到原始数据中
    if new_rows:
        # 将新行转换为DataFrame并附加到原始数据上
        augmented_df = pd.DataFrame(new_rows)
        result_data = pd.concat(
            [augmented_data, augmented_df], ignore_index=True)

        # 添加is_augmented列，原始数据标记为0
        if "is_augmented" not in result_data.columns:
            result_data["is_augmented"] = 0
            # 标记新增行为1
            result_data.loc[len(augmented_data):, "is_augmented"] = 1
    else:
        result_data = augmented_data
        result_data["is_augmented"] = 0

    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir, f"augmented_room_bed_data_{timestamp}.csv")

    # 保存结果
    print(f"\n正在保存处理结果...")
    result_data.to_csv(output_file, index=False, encoding="utf-8")
    logger.info(f"处理结果已保存到 {output_file}")

    # 打印统计信息
    logger.info(f"原始数据行数: {len(data)}")
    logger.info(f"增强后数据行数: {len(result_data)}")
    logger.info(f"新增数据行数: {len(new_rows)}")

    # 输出前5个增强示例
    if new_rows:
        logger.info("\n增强数据示例:")
        for i in range(min(5, len(new_rows))):
            orig_idx = selected_records[i][0]
            original_text = data.at[orig_idx, "spl_room_text"]
            augmented_text = new_rows[i]["spl_room_text"]
            logger.info(f"原始文本: {original_text}")
            logger.info(f"增强文本: {augmented_text}")
            logger.info("-----")

    print(
        f"\n处理完成！原始数据 {len(data)} 条，增强后 {len(result_data)} 条，新增 {len(new_rows)} 条")
    return output_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="床位数据增强处理")
    parser.add_argument("--input_file", type=str, default=CONFIG["input_file"],
                        help="输入CSV文件路径")
    parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"],
                        help="输出目录")
    parser.add_argument("--augmentation_ratio", type=float, default=CONFIG["augmentation_ratio"],
                        help="数据增强比例，范围0-1，默认0.05（5%）")
    parser.add_argument("--random_seed", type=int, default=CONFIG["random_seed"],
                        help="随机种子，用于结果复现")

    args = parser.parse_args()

    try:
        # 确保logs目录存在
        log_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 记录开始时间
        start_time = datetime.now()
        print(f"开始床位数据增强处理: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        output_file = augment_bed_data(
            args.input_file,
            args.output_dir,
            args.augmentation_ratio,
            args.random_seed
        )

        # 记录结束时间及总耗时
        end_time = datetime.now()
        duration = end_time - start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"总耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")

        if output_file:
            logger.info(f"处理完成。结果保存在: {output_file}")
            print(f"处理完成。结果保存在: {output_file}")
        else:
            logger.warning("处理过程未生成结果文件")
            print("处理过程未生成结果文件")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        print(f"处理过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
