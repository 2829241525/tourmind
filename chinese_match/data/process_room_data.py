#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
处理房间数据CSV脚本
- 读取指定列
- 合并房间名称和床型信息
- 输出新的CSV文件
"""

import os
import pandas as pd
from typing import Optional
import logging

# 全局配置项
config = {
    # 输入CSV文件路径，使用绝对路径
    "input_file": "/home/maxon/disk2/roomMatch/room_match/chinese_match/data/all_cn_data.csv",

    # 输出CSV文件路径，如果不指定则自动生成
    "output_file": None,

    # 必要列名称列表
    "required_columns": [
        'spl_room_id', 'supplier_id', 'supplier_name', 's_hotel_id',
        'spl_room_name', 'spl_room_bed_name', 's_room_name', 's_room_bed_name',
        'label', 'similar'
    ],

    # 输出列名称列表
    "output_columns": [
        'spl_room_id', 'supplier_id', 'supplier_name', 's_hotel_id',
        'spl_room_text', 's_room_text', 'label', 'similar'
    ],

    # 列合并时使用的分隔符
    "join_delimiter": ",",

    # 是否过滤掉房间名为空的行
    "filter_empty_room_name": True
}

# 如果没有指定输出文件路径，根据输入文件名自动生成
if config["output_file"] is None:
    input_basename = os.path.basename(config["input_file"])
    input_name, _ = os.path.splitext(input_basename)
    config["output_file"] = os.path.join(os.path.dirname(
        config["input_file"]), f"{input_name}_processed.csv")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_room_data(
    input_file: str,
    output_file: Optional[str] = None,
    required_columns: Optional[list] = None,
    output_columns: Optional[list] = None,
    join_delimiter: str = ",",
    filter_empty_room_name: bool = True
) -> pd.DataFrame:
    """
    处理房间数据

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径，如果为None则不保存
        required_columns: 需要读取的列
        output_columns: 需要输出的列
        join_delimiter: 合并列时使用的分隔符
        filter_empty_room_name: 是否过滤掉房间名为空的行

    Returns:
        pd.DataFrame: 处理后的数据
    """
    # 使用默认值如果未提供参数
    if required_columns is None:
        required_columns = config["required_columns"]

    if output_columns is None:
        output_columns = config["output_columns"]

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到输入文件：{input_file}")

    # 读取CSV文件
    logger.info(f"正在读取CSV文件：{input_file}")
    try:
        # 尝试读取CSV文件
        df = pd.read_csv(input_file, usecols=lambda x: x in required_columns)

        # 检查必要列是否存在
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"以下列在CSV文件中不存在，将被创建为空列：{missing_columns}")
            for col in missing_columns:
                df[col] = pd.NA
    except Exception as e:
        logger.error(f"读取CSV文件失败: {str(e)}")
        # 尝试不过滤列读取
        logger.info("正在尝试不过滤列读取文件...")
        df = pd.read_csv(input_file)

        # 确保必要列存在
        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA

    # 计算数据行数
    row_count = len(df)
    logger.info(f"读取了 {row_count} 行数据")

    # 处理缺失值
    df = df.fillna("")

    # 过滤掉房间名为空的行
    if filter_empty_room_name:
        original_count = len(df)
        df = df[(df['spl_room_name'].str.strip() != "")
                & (df['s_room_name'].str.strip() != "")]
        filtered_count = original_count - len(df)
        logger.info(f"已过滤掉 {filtered_count} 行房间名为空的数据")

    # 合并列并创建新列
    logger.info("正在合并列...")
    df['spl_room_text'] = df.apply(
        lambda row: _join_values(
            row['spl_room_name'], row['spl_room_bed_name'], join_delimiter),
        axis=1
    )
    df['s_room_text'] = df.apply(
        lambda row: _join_values(
            row['s_room_name'], row['s_room_bed_name'], join_delimiter),
        axis=1
    )

    # 确保所有输出列存在
    for col in output_columns:
        if col not in df.columns:
            df[col] = ""

    # 只保留需要的列
    result_df = df[output_columns]

    # 保存结果
    if output_file:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info(f"正在保存结果到：{output_file}")
        result_df.to_csv(output_file, index=False)
        logger.info(f"已保存 {len(result_df)} 行数据到 {output_file}")

    return result_df


def _join_values(value1: str, value2: str, delimiter: str) -> str:
    """
    合并两个值，如果一方为空，则不使用分隔符

    Args:
        value1: 第一个值
        value2: 第二个值
        delimiter: 分隔符

    Returns:
        str: 合并后的字符串
    """
    value1 = value1.strip() if value1 else ""
    value2 = value2.strip() if value2 else ""

    if not value1:
        return value2
    if not value2:
        return value1
    return f"{value1}{delimiter}{value2}"


def main():
    """主函数"""
    # 使用全局配置进行处理
    process_room_data(
        input_file=config["input_file"],
        output_file=config["output_file"],
        required_columns=config["required_columns"],
        output_columns=config["output_columns"],
        join_delimiter=config["join_delimiter"],
        filter_empty_room_name=config.get("filter_empty_room_name", True)
    )

    logger.info("处理完成！")


if __name__ == "__main__":
    main()
