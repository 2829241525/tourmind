#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
from datetime import datetime

# 配置参数
config = {
    "input_csv_path": "/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group_clean.csv",
    "output_csv_path": "/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group_clean_db.csv",
    "log_level": "INFO"
}


def setup_logger():
    """设置日志"""
    import logging
    logging.basicConfig(
        level=getattr(logging, config["log_level"]),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = setup_logger()


def process_csv():
    """
    处理CSV文件：
    1. 读取原始CSV
    2. 复制数据并置换spl_room_text和s_room_text列
    3. 拼接原始数据和修改后的数据
    4. 输出到新的CSV文件
    """
    try:
        logger.info(f"开始读取CSV文件: {config['input_csv_path']}")
        # 检查输入文件是否存在
        if not os.path.exists(config["input_csv_path"]):
            logger.error(f"输入文件不存在: {config['input_csv_path']}")
            return False

        # 读取原始CSV文件
        df = pd.read_csv(config["input_csv_path"])
        logger.info(f"成功读取CSV文件，共 {len(df)} 行数据")

        # 复制数据并交换spl_room_text和s_room_text列
        df_swapped = df.copy()

        # 检查必要的列是否存在
        required_cols = ['spl_room_text', 's_room_text']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [
                col for col in required_cols if col not in df.columns]
            logger.error(f"CSV文件缺少必要的列: {missing_cols}")
            return False

        # 交换两列的值
        df_swapped['spl_room_text'], df_swapped['s_room_text'] = \
            df_swapped['s_room_text'].copy(
        ), df_swapped['spl_room_text'].copy()

        logger.info(f"已完成列置换，准备合并数据")

        # 合并原始数据和修改后的数据
        df_combined = pd.concat([df, df_swapped], ignore_index=True)

        logger.info(f"数据合并完成，合并后共 {len(df_combined)} 行数据")

        # 保存到新CSV文件
        output_dir = os.path.dirname(config["output_csv_path"])
        if not os.path.exists(output_dir) and output_dir:
            os.makedirs(output_dir)

        df_combined.to_csv(config["output_csv_path"], index=False)
        logger.info(f"处理完成，输出文件保存至: {config['output_csv_path']}")

        return True

    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}")
        return False


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"开始执行CSV处理程序...")

    success = process_csv()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    if success:
        logger.info(f"程序执行成功，耗时: {duration:.2f} 秒")
    else:
        logger.error(f"程序执行失败，耗时: {duration:.2f} 秒")
