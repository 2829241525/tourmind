#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV输出功能测试脚本
测试大模型匹配结果的CSV输出功能
"""

from steps import HttpInfoProcessor
from room_matching_workflow import RoomMatchingWorkflow
import json
import logging
import sys
import os
import csv

# 添加父目录到路径以支持直接运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_csv_output_function():
    """测试CSV输出功能"""
    logger.info("=== CSV输出功能测试 ===")

    # 创建工作流实例
    workflow = RoomMatchingWorkflow()

    # 测试不同的酒店ID
    test_hotels = [
        {"hotel_id": "2146988", "description": "测试酒店1"},
        {"hotel_id": "799311", "description": "测试酒店2"},
        {"hotel_id": "8789588", "description": "测试酒店3"},
    ]

    success_count = 0
    total_count = len(test_hotels)
    csv_files = []

    for i, hotel_info in enumerate(test_hotels, 1):
        hotel_id = hotel_info["hotel_id"]
        description = hotel_info["description"]
        
        print(f"\n{'='*80}")
        print(f"测试 {i}/{total_count}: {description} (酒店ID: {hotel_id})")
        print(f"{'='*80}")

        try:
            # 执行步骤1
            result = workflow.execute_single_step(1, hotel_id=hotel_id, method="GET")

            if result.get('status') == 'success':
                print(f"✅ {description} 处理成功")
                
                # 获取CSV输出信息
                csv_output = result.get('csv_output', {})
                csv_filepath = csv_output.get('filepath', '')
                parsed_matches_count = csv_output.get('parsed_matches_count', 0)
                
                if csv_filepath and os.path.exists(csv_filepath):
                    print(f"📄 CSV文件已生成: {csv_filepath}")
                    print(f"📊 解析的匹配关系数量: {parsed_matches_count}")
                    
                    # 验证CSV文件内容
                    try:
                        with open(csv_filepath, 'r', encoding='utf-8') as f:
                            csv_reader = csv.reader(f)
                            rows = list(csv_reader)
                            
                            if rows:
                                headers = rows[0]
                                data_rows = rows[1:]
                                
                                print(f"📋 CSV文件结构:")
                                print(f"   表头: {headers}")
                                print(f"   数据行数: {len(data_rows)}")
                                
                                # 显示前3行数据作为示例
                                if data_rows:
                                    print(f"   示例数据:")
                                    for j, row in enumerate(data_rows[:3], 1):
                                        print(f"     [{j}] {row}")
                                
                                csv_files.append({
                                    'hotel_id': hotel_id,
                                    'filepath': csv_filepath,
                                    'rows': len(data_rows),
                                    'matches': parsed_matches_count
                                })
                                
                                success_count += 1
                            else:
                                print(f"❌ CSV文件为空")
                                
                    except Exception as e:
                        print(f"❌ 读取CSV文件失败: {str(e)}")
                        
                else:
                    print(f"❌ 未生成CSV文件或文件不存在")
                    
            else:
                error_msg = result.get('error', '未知错误')
                print(f"❌ {description} 处理失败: {error_msg}")

        except Exception as e:
            print(f"❌ {description} 执行异常: {str(e)}")

        print(f"{'='*80}")

    # 输出测试总结
    print(f"\n🎯 CSV输出功能测试总结:")
    print(f"   总测试数量: {total_count}")
    print(f"   成功数量: {success_count}")
    print(f"   成功率: {success_count/total_count*100:.1f}%")
    
    if csv_files:
        print(f"\n📁 生成的CSV文件列表:")
        for i, file_info in enumerate(csv_files, 1):
            print(f"   [{i}] 酒店ID {file_info['hotel_id']}: {file_info['filepath']}")
            print(f"       数据行数: {file_info['rows']}, 匹配数: {file_info['matches']}")

    return csv_files


def analyze_csv_files(csv_files):
    """分析生成的CSV文件"""
    if not csv_files:
        print("❌ 没有CSV文件可分析")
        return

    print(f"\n🔍 CSV文件分析:")
    print(f"{'='*80}")
    
    total_rows = 0
    total_matches = 0
    
    for file_info in csv_files:
        filepath = file_info['filepath']
        hotel_id = file_info['hotel_id']
        
        print(f"\n📄 分析文件: {os.path.basename(filepath)}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                rows = list(csv_reader)
                
                if len(rows) > 1:  # 有数据行
                    headers = rows[0]
                    data_rows = rows[1:]
                    
                    # 统计匹配状态
                    matched_count = 0
                    unmatched_count = 0
                    
                    for row in data_rows:
                        if len(row) >= 5:  # 确保有匹配状态列
                            status = row[4].strip()
                            if status == "已匹配":
                                matched_count += 1
                            else:
                                unmatched_count += 1
                    
                    total_rows += len(data_rows)
                    total_matches += matched_count
                    
                    print(f"   酒店ID: {hotel_id}")
                    print(f"   总房型数: {len(data_rows)}")
                    print(f"   已匹配: {matched_count}")
                    print(f"   未匹配: {unmatched_count}")
                    print(f"   匹配率: {matched_count/len(data_rows)*100:.1f}%")
                    
        except Exception as e:
            print(f"   ❌ 分析失败: {str(e)}")
    
    print(f"\n📊 总体统计:")
    print(f"   总房型数: {total_rows}")
    print(f"   总匹配数: {total_matches}")
    print(f"   总体匹配率: {total_matches/total_rows*100:.1f}%" if total_rows > 0 else "   总体匹配率: 0%")
    print(f"{'='*80}")


if __name__ == "__main__":
    print("CSV输出功能测试")
    print("="*80)

    # 运行CSV输出功能测试
    csv_files = test_csv_output_function()
    
    # 分析生成的CSV文件
    analyze_csv_files(csv_files)

    print(f"\n🎉 CSV输出功能测试完成!") 