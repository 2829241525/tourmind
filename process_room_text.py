#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import logging
import os

# 配置信息
CONFIG = {
    'input_file': os.path.abspath('input.csv'),  # 输入CSV文件路径
    'output_file': os.path.abspath('output.csv'),  # 输出CSV文件路径
    'target_columns': ['spl_room_text', 's_room_text'],  # 需要处理的列名
    'log_file': os.path.abspath('process_room.log')  # 日志文件路径
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_file'], encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def process_room_text(text):
    """
    处理房间文本
    1. 移除所有[room]标记
    2. 如果包含[bed][unknown]，则移除这些标记
    3. 如果只包含[bed]，则替换为逗号
    """
    if not isinstance(text, str):
        return text

    # 移除所有[room]标记
    text = text.replace('[room]', '')

    # 检查是否包含[bed][unknown]
    if '[bed]' in text and '[unknown]' in text:
        text = text.replace('[bed]', '').replace('[unknown]', '')
    # 如果只包含[bed]，替换为逗号
    elif '[bed]' in text:
        text = text.replace('[bed]', ',')

    return text.strip()


def main():
    try:
        # 检查输入文件是否存在
        if not os.path.exists(CONFIG['input_file']):
            logging.error(f"输入文件不存在: {CONFIG['input_file']}")
            return

        # 读取CSV文件
        logging.info(f"开始读取CSV文件: {CONFIG['input_file']}")
        df = pd.read_csv(CONFIG['input_file'])

        # 检查目标列是否存在
        missing_columns = [
            col for col in CONFIG['target_columns'] if col not in df.columns]
        if missing_columns:
            logging.error(f"以下列在CSV中不存在: {missing_columns}")
            return

        # 处理目标列
        for column in CONFIG['target_columns']:
            logging.info(f"正在处理列: {column}")
            df[column] = df[column].apply(process_room_text)

        # 保存处理后的文件
        df.to_csv(CONFIG['output_file'], index=False)
        logging.info(f"处理完成，已保存到: {CONFIG['output_file']}")

    except Exception as e:
        logging.error(f"处理过程中发生错误: {str(e)}")


if __name__ == "__main__":
    main()
