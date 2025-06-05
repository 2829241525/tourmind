import pandas as pd
import os
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置路径
DATA_DIR = Path('/home/maxon/disk2/roomMatch/room_match/data')
INPUT_FILE = DATA_DIR / 'room_samples.csv'
OUTPUT_FILE = DATA_DIR / 'room_samples_processed_text_pairs.csv'


def process_text_pairs():
    try:
        # 读取原始CSV文件
        logging.info(f"正在读取文件: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)

        # 统计原始数据的正负样本比例
        label_counts = df['label'].value_counts()
        total_samples = len(df)
        logging.info(f"原始数据正负样本统计：")
        logging.info(f"总样本数: {total_samples}")
        for label, count in label_counts.items():
            percentage = (count / total_samples) * 100
            logging.info(f"标签 {label}: {count} 个样本 ({percentage:.2f}%)")

        # 创建交换text1和text2的新数据框
        df_swapped = df.copy()
        df_swapped['text1'], df_swapped['text2'] = df_swapped['text2'], df_swapped['text1']

        # 合并原始数据和交换后的数据
        df_combined = pd.concat([df, df_swapped], ignore_index=True)

        # 去重，保持第一次出现的记录
        df_unique = df_combined.drop_duplicates(
            subset=['text1', 'text2'], keep='first')

        # 统计去重后的正负样本比例
        unique_label_counts = df_unique['label'].value_counts()
        unique_total = len(df_unique)
        logging.info(f"\n去重后数据正负样本统计：")
        logging.info(f"总样本数: {unique_total}")
        for label, count in unique_label_counts.items():
            percentage = (count / unique_total) * 100
            logging.info(f"标签 {label}: {count} 个样本 ({percentage:.2f}%)")

        # 保存处理后的数据
        logging.info(f"正在保存处理后的数据到: {OUTPUT_FILE}")
        df_unique.to_csv(OUTPUT_FILE, index=False)

        logging.info(f"处理完成。原始数据行数: {len(df)}, 处理后数据行数: {len(df_unique)}")

    except Exception as e:
        logging.error(f"处理过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    process_text_pairs()
