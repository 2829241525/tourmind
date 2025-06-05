#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将Excel文件(xlsx)转换为CSV文件
"""

import os
import logging
import argparse
import pandas as pd
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('xlsx_to_csv')

# 默认配置
DEFAULT_DATA_DIR = '/home/maxon/disk2/roomMatch/room_match/deberta/data'
DEFAULT_ENCODING = 'utf-8'
DEFAULT_INDEX = False


def xlsx_to_csv(xlsx_file, output_file=None, encoding='utf-8', index=False):
    """
    将单个xlsx文件转换为csv文件
    
    参数:
        xlsx_file (str): xlsx文件的路径
        output_file (str): 输出csv文件的路径，如果为None，则使用相同的文件名但扩展名为csv
        encoding (str): 输出csv文件的编码
        index (bool): 是否保留DataFrame的索引
    
    返回:
        bool: 转换是否成功
    """
    try:
        xlsx_path = Path(xlsx_file)
        if not xlsx_path.exists():
            logger.error(f"文件不存在: {xlsx_file}")
            return False
        
        # 如果没有指定输出文件，则使用相同的文件名但扩展名为csv
        if output_file is None:
            output_file = xlsx_path.with_suffix('.csv')
        
        logger.info(f"正在将 {xlsx_file} 转换为 {output_file}")
        
        # 读取xlsx文件
        df = pd.read_excel(xlsx_file)
        
        # 保存为csv文件
        df.to_csv(output_file, encoding=encoding, index=index)
        
        logger.info(f"转换成功: {output_file}")
        return True
    except Exception as e:
        logger.error(f"转换失败: {e}")
        return False


def batch_convert(directory, encoding=DEFAULT_ENCODING, index=DEFAULT_INDEX):
    """
    批量转换目录中的所有xlsx文件
    
    参数:
        directory (str): 包含xlsx文件的目录
        encoding (str): 输出csv文件的编码
        index (bool): 是否保留DataFrame的索引
    
    返回:
        int: 成功转换的文件数量
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        logger.error(f"目录不存在: {directory}")
        return 0
    
    success_count = 0
    for xlsx_file in directory.glob('*.xlsx'):
        if xlsx_to_csv(xlsx_file, encoding=encoding, index=index):
            success_count += 1
    
    logger.info(f"批量转换完成，成功转换 {success_count} 个文件")
    return success_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将Excel文件(xlsx)转换为CSV文件')
    parser.add_argument('-d', '--directory', type=str, default=DEFAULT_DATA_DIR,
                        help=f'包含xlsx文件的目录 (默认: {DEFAULT_DATA_DIR})')
    parser.add_argument('-f', '--file', type=str, help='单个xlsx文件的路径')
    parser.add_argument('-o', '--output', type=str, help='输出csv文件的路径 (仅适用于单个文件转换)')
    parser.add_argument('-e', '--encoding', type=str, default=DEFAULT_ENCODING,
                        help=f'输出csv文件的编码 (默认: {DEFAULT_ENCODING})')
    parser.add_argument('--index', action='store_true', default=DEFAULT_INDEX,
                        help='是否保留DataFrame的索引 (默认: False)')
    
    args = parser.parse_args()
    
    if args.file:
        # 转换单个文件
        xlsx_to_csv(args.file, args.output, args.encoding, args.index)
    else:
        # 批量转换目录中的所有xlsx文件
        batch_convert(args.directory, args.encoding, args.index)


if __name__ == '__main__':
    main() 