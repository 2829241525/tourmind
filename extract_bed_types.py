#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
床型信息提取工具

这个脚本用于从CSV文件中提取床型信息，将提取到的床型ID和描述添加到原始数据中。
"""

import os
import sys
import argparse
import time
from datetime import datetime

# 确保可以导入sequence_annotation包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sequence_annotation.bed_conversion import extract_bed_type, with_beds
except ImportError:
    print("错误: 无法导入bed_conversion模块，请确保脚本位置正确")
    sys.exit(1)

def process_csv(input_file, output_file, column_name):
    """
    从CSV文件中提取床型信息
    
    Args:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
        column_name (str): 需要处理的列名
    
    Returns:
        tuple: (总行数, 含床型信息行数)
    """
    import pandas as pd
    import logging
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger()
    
    print(f"开始处理CSV文件: {input_file}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return 0, 0
    
    try:
        # 读取CSV文件
        # 使用较低的内存占用方式读取大文件
        chunk_size = 10000  # 每次读取的行数
        total_rows = 0
        processed_rows = 0
        has_bed_rows = 0
        
        # 创建输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
        
        # 分块处理CSV
        for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, encoding='utf-8', low_memory=False)):
            print(f"正在处理第{chunk_idx+1}块数据，{chunk_size}行")
            
            # 检查列是否存在
            if column_name not in chunk.columns:
                print(f"错误: 列名 {column_name} 不存在于CSV文件中")
                return 0, 0
            
            # 添加新列
            chunk['bed_id'] = None
            chunk['bed_description'] = None
            
            # 处理每一行
            for idx, row in chunk.iterrows():
                total_rows += 1
                room_description = row[column_name]
                
                # 检查是否为空值
                if pd.isna(room_description) or not isinstance(room_description, str):
                    continue
                
                # 检查是否包含床型描述
                if with_beds(room_description):
                    # 提取床型信息
                    bed_id, bed_desc = extract_bed_type(room_description)
                    chunk.at[idx, 'bed_id'] = bed_id
                    chunk.at[idx, 'bed_description'] = bed_desc
                    has_bed_rows += 1
                
                processed_rows += 1
                
                # 打印进度
                if processed_rows % 1000 == 0:
                    print(f"已处理 {processed_rows} 行，发现 {has_bed_rows} 行含有床型信息")
            
            # 将处理后的数据追加到输出文件
            chunk.to_csv(output_file, mode='a', index=False, header=(chunk_idx==0), encoding='utf-8')
            
        print(f"处理完成! 总共处理了 {total_rows} 行，发现 {has_bed_rows} 行含有床型信息")
        print(f"结果已保存至: {output_file}")
        
        return total_rows, has_bed_rows
    
    except Exception as e:
        import traceback
        print(f"处理CSV时发生错误: {str(e)}")
        print(traceback.format_exc())
        return 0, 0

def test_extraction():
    """简单测试床型提取功能"""
    # 测试用例
    test_cases = [
        "Deluxe Room with 1 King Bed",
        "Standard Room with 2 Twin Beds",
        "豪华大床房",
        "标准双床房",
        "家庭房带上下铺"
    ]
    
    print("\n===== 床型提取测试 =====")
    for case in test_cases:
        if with_beds(case):
            bed_id, bed_desc = extract_bed_type(case)
            print(f"{case}: [ID={bed_id}, 描述={bed_desc}]")
        else:
            print(f"{case}: 未找到床型信息")

def main():
    """主程序入口"""
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='床型信息提取工具')
    
    parser.add_argument('-i', '--input', 
                      type=str, 
                      default='/home/maxon/disk2/roomMatch/room_match/data/bed_all_data_process.csv',
                      help='输入CSV文件路径')
    
    parser.add_argument('-o', '--output', 
                      type=str, 
                      default='/home/maxon/disk2/roomMatch/room_match/data/bed_all_data_with_type.csv',
                      help='输出CSV文件路径')
    
    parser.add_argument('-c', '--column', 
                      type=str, 
                      default='spl_cleaned_room',
                      help='需要处理的列名')
    
    parser.add_argument('-t', '--test', 
                      action='store_true',
                      help='运行简单测试')
    
    args = parser.parse_args()
    
    # 简单测试
    if args.test:
        test_extraction()
        return
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在 - {args.input}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 处理CSV文件
    total_rows, has_bed_rows = process_csv(args.input, args.output, args.column)
    
    # 记录结束时间并输出统计
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n运行统计:")
    print(f"总行数: {total_rows}")
    print(f"含床型信息行数: {has_bed_rows}")
    
    if total_rows > 0:
        print(f"提取成功率: {has_bed_rows/total_rows*100:.2f}%")
    
    print(f"总运行时间: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")

if __name__ == "__main__":
    print("""
===================================
床型信息提取工具 v1.0
===================================
""")
    main() 