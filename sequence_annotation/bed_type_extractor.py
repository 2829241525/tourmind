# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import os
# import sys
# import pandas as pd
# import logging
# import time
# import argparse
# from datetime import datetime

# # 尝试导入bed_conversion模块，处理不同的导入情况
# try:
#     # 如果脚本在sequence_annotation目录内运行
#     from bed_conversion import extract_bed_type, with_beds
#     print("使用直接导入方式")
# except ImportError:
#     try:
#         # 如果从父目录运行
#         from sequence_annotation.bed_conversion import extract_bed_type, with_beds
#         print("使用包导入方式")
#     except ImportError:
#         try:
#             # 如果在同一目录下，使用相对导入
#             from .bed_conversion import extract_bed_type, with_beds
#             print("使用相对导入方式")
#         except (ImportError, ValueError):
#             print("错误: 无法导入bed_conversion模块，请确保脚本位置正确")
#             sys.exit(1)

# # 配置日志
# LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
# os.makedirs(LOG_DIR, exist_ok=True)
# log_file = os.path.join(LOG_DIR, f"bed_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(log_file, encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # 默认配置
# DEFAULT_CONFIG = {
#     'input_file': '/home/maxon/disk2/roomMatch/room_match/data/bed_all_data_process.csv',
#     'output_file': '/home/maxon/disk2/roomMatch/room_match/data/bed_all_data_with_type.csv',
#     'column_name': 'spl_cleaned_room'
# }

# def process_csv(config=DEFAULT_CONFIG):
#     """
#     处理CSV文件，提取床型信息
    
#     Args:
#         config (dict): 配置信息，包含输入文件路径、输出文件路径、列名
#     """
#     input_file = config['input_file']
#     output_file = config['output_file']
#     column_name = config['column_name']
    
#     logger.info(f"开始处理CSV文件: {input_file}")
    
#     # 检查输入文件是否存在
#     if not os.path.exists(input_file):
#         logger.error(f"输入文件不存在: {input_file}")
#         return 0, 0
    
#     try:
#         # 读取CSV文件
#         # 使用较低的内存占用方式读取大文件
#         chunk_size = 10000  # 每次读取的行数
#         total_rows = 0
#         processed_rows = 0
#         has_bed_rows = 0
        
#         # 创建输出文件
#         with open(output_file, 'w', encoding='utf-8') as f:
#             pass
        
#         # 分块处理CSV
#         for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, encoding='utf-8', low_memory=False)):
#             logger.info(f"正在处理第{chunk_idx+1}块数据，{chunk_size}行")
            
#             # 检查列是否存在
#             if column_name not in chunk.columns:
#                 logger.error(f"列名 {column_name} 不存在于CSV文件中")
#                 return 0, 0
            
#             # 添加新列
#             chunk['bed_id'] = None
#             chunk['bed_description'] = None
            
#             # 处理每一行
#             for idx, row in chunk.iterrows():
#                 total_rows += 1
#                 room_description = row[column_name]
                
#                 # 检查是否为空值
#                 if pd.isna(room_description) or not isinstance(room_description, str):
#                     continue
                
#                 # 检查是否包含床型描述
#                 if with_beds(room_description):
#                     # 提取床型信息
#                     bed_id, bed_desc = extract_bed_type(room_description)
#                     chunk.at[idx, 'bed_id'] = bed_id
#                     chunk.at[idx, 'bed_description'] = bed_desc
#                     has_bed_rows += 1
                
#                 processed_rows += 1
                
#                 # 打印进度
#                 if processed_rows % 1000 == 0:
#                     logger.info(f"已处理 {processed_rows} 行，发现 {has_bed_rows} 行含有床型信息")
            
#             # 将处理后的数据追加到输出文件
#             chunk.to_csv(output_file, mode='a', index=False, header=(chunk_idx==0), encoding='utf-8')
            
#         logger.info(f"处理完成! 总共处理了 {total_rows} 行，发现 {has_bed_rows} 行含有床型信息")
#         logger.info(f"结果已保存至: {output_file}")
        
#         return total_rows, has_bed_rows
    
#     except Exception as e:
#         logger.exception(f"处理CSV时发生错误: {str(e)}")
#         return 0, 0

# def parse_args():
#     """
#     解析命令行参数
    
#     Returns:
#         args: 命令行参数对象
#     """
#     parser = argparse.ArgumentParser(description='床型信息提取工具')
    
#     parser.add_argument('-i', '--input', 
#                       dest='input_file',
#                       type=str, 
#                       default=DEFAULT_CONFIG['input_file'],
#                       help=f"输入CSV文件路径 (默认: {DEFAULT_CONFIG['input_file']})")
    
#     parser.add_argument('-o', '--output', 
#                       dest='output_file',
#                       type=str, 
#                       default=DEFAULT_CONFIG['output_file'],
#                       help=f"输出CSV文件路径 (默认: {DEFAULT_CONFIG['output_file']})")
    
#     parser.add_argument('-c', '--column', 
#                       dest='column_name',
#                       type=str, 
#                       default=DEFAULT_CONFIG['column_name'],
#                       help=f"需要处理的列名 (默认: {DEFAULT_CONFIG['column_name']})")
    
#     parser.add_argument('-t', '--test', 
#                       action='store_true',
#                       help='运行测试床型提取')
    
#     parser.add_argument('-v', '--verbose', 
#                       action='store_true',
#                       help='显示详细输出')
    
#     args = parser.parse_args()
    
#     # 检查输入文件是否存在
#     if not args.test and not os.path.exists(args.input_file):
#         print(f"错误: 输入文件不存在 - {args.input_file}")
#         sys.exit(1)
    
#     # 检查输出目录是否存在
#     if not args.test:
#         output_dir = os.path.dirname(args.output_file)
#         if output_dir and not os.path.exists(output_dir):
#             os.makedirs(output_dir, exist_ok=True)
    
#     return args

# def test_bed_extraction():
#     """
#     测试床型提取功能
#     """
#     # 测试用例
#     test_cases = [
#         # 英文床型测试
#         "Deluxe Room with 1 King Bed",
#         "Standard Room with 2 Twin Beds",
#         "Suite with 1 Queen Bed and 1 Sofa Bed",
#         "Family Room with 1 Double Bed and 2 Single Beds",
#         "Room with Bunk Bed",
#         "Semi-Double Room",
#         # 中文床型测试
#         "豪华大床房",
#         "标准双床房",
#         "家庭房带上下铺",
#         # 更多测试用例
#         "1 Queen Bed",
#         "2 Twin Beds",
#         "1 King or 2 Single Beds",
#         "1 Double Bed or 2 Single Beds",
#         "1 Full Bed",
#         "City View Room with 1 king bed",
#         "Standard Double or Twin Room",
#         "Double Room with 1 double bed",
#         "Queen Room - Non-Smoking",
#         "Superior Double",
#     ]
    
#     print("\n===== 床型提取测试 =====")
#     print("=" * 60)
#     print(f"{'房型描述':<40} | {'床型ID':<6} | {'床型描述':<20}")
#     print("-" * 60)
    
#     for case in test_cases:
#         # 检查是否含有床型描述
#         if with_beds(case):
#             # 提取床型ID和描述
#             bed_id, bed_desc = extract_bed_type(case)
#             print(f"{case[:38]:<40} | {bed_id:<6} | {bed_desc:<20}")
#         else:
#             print(f"{case[:38]:<40} | {'N/A':<6} | {'无床型信息':<20}")
    
#     print("=" * 60)

# def create_test_csv():
#     """
#     创建测试用CSV文件
#     """
#     # 测试数据
#     test_data = {
#         'id': range(1, 21),
#         'hotel_name': [f'Hotel {i}' for i in range(1, 21)],
#         'spl_cleaned_room': [
#             "Deluxe Room with 1 King Bed",
#             "Standard Room with 2 Twin Beds",
#             "Suite with 1 Queen Bed and 1 Sofa Bed",
#             "Family Room with 1 Double Bed and 2 Single Beds",
#             "Room with Bunk Bed",
#             "Semi-Double Room",
#             "豪华大床房",
#             "标准双床房",
#             "家庭房带上下铺",
#             "1 Queen Bed",
#             "2 Twin Beds",
#             "1 King or 2 Single Beds",
#             "1 Double Bed or 2 Single Beds",
#             "1 Full Bed",
#             "City View Room with 1 king bed",
#             "Standard Double or Twin Room",
#             "Double Room with 1 double bed",
#             "Queen Room - Non-Smoking",
#             "Superior Double",
#             "Standard Room" # 无床型信息
#         ]
#     }
    
#     # 创建DataFrame
#     df = pd.DataFrame(test_data)
    
#     # 保存为CSV
#     test_csv_path = "test_bed_data.csv"
#     df.to_csv(test_csv_path, index=False, encoding='utf-8')
    
#     print(f"\n测试CSV文件已创建: {os.path.abspath(test_csv_path)}")
#     return test_csv_path

# def test_with_csv():
#     """
#     使用测试CSV进行床型提取测试
#     """
#     # 创建测试CSV
#     test_csv_path = create_test_csv()
    
#     # 设置输出路径
#     output_csv_path = "test_bed_result.csv"
    
#     # 处理测试CSV
#     config = {
#         'input_file': test_csv_path,
#         'output_file': output_csv_path,
#         'column_name': 'spl_cleaned_room'
#     }
    
#     print("\n开始处理测试CSV文件...")
#     process_csv(config)
    
#     # 读取结果
#     if os.path.exists(output_csv_path):
#         df_result = pd.read_csv(output_csv_path, encoding='utf-8')
        
#         print("\n===== 处理结果 =====")
#         print(df_result[['id', 'spl_cleaned_room', 'bed_id', 'bed_description']].to_string(index=False))
        
#         # 统计
#         total_rows = len(df_result)
#         has_bed_rows = df_result['bed_id'].notna().sum()
        
#         print(f"\n总行数: {total_rows}")
#         print(f"含床型信息行数: {has_bed_rows}")
#         print(f"提取成功率: {has_bed_rows/total_rows*100:.2f}%")
#     else:
#         print(f"输出文件未找到: {output_csv_path}")

# def main():
#     """程序主入口"""
#     # 解析命令行参数
#     args = parse_args()
    
#     # 如果是测试模式
#     if args.test:
#         logger.info("运行测试模式")
#         test_bed_extraction()
#         test_with_csv()
#         return
    
#     # 构建配置
#     config = {
#         'input_file': args.input_file,
#         'output_file': args.output_file,
#         'column_name': args.column_name
#     }
    
#     logger.info("配置信息:")
#     logger.info(f"输入文件: {config['input_file']}")
#     logger.info(f"输出文件: {config['output_file']}")
#     logger.info(f"处理列名: {config['column_name']}")
    
#     # 记录开始时间
#     start_time = time.time()
    
#     # 处理CSV文件
#     total_rows, has_bed_rows = process_csv(config)
    
#     # 记录结束时间
#     end_time = time.time()
#     elapsed_time = end_time - start_time
    
#     # 打印运行时间和统计信息
#     logger.info(f"总运行时间: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    
#     if total_rows > 0:
#         success_rate = has_bed_rows/total_rows*100
#         logger.info(f"提取成功率: {success_rate:.2f}%")

# if __name__ == "__main__":
#     print("""
# ===================================
# 床型信息提取工具 v1.0
# ===================================

# 功能:
#   从CSV文件中提取床型信息，并将结果输出到新的CSV文件

# 使用方法:
#   python bed_type_extractor.py [选项]

# 选项:
#   -i, --input    输入CSV文件路径
#   -o, --output   输出CSV文件路径
#   -c, --column   需要处理的列名
#   -t, --test     运行测试
#   -v, --verbose  显示详细信息
#   -h, --help     显示帮助信息

# 示例:
#   python bed_type_extractor.py
#   python bed_type_extractor.py -i data/my_rooms.csv -o data/rooms_with_bed.csv
#   python bed_type_extractor.py -t
# """)
#     main() 