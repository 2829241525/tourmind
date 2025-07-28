#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房型匹配工作流 - 演示脚本
专注于测试HTTP信息处理器的规则抽取功能
"""

from steps import HttpInfoProcessor
from room_matching_workflow import RoomMatchingWorkflow
import json
import logging
import sys
import os

# 添加父目录到路径以支持直接运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 尝试绝对导入


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_http_info_processor():
    """演示HTTP信息处理器的规则抽取功能"""
    logger.info("=== HTTP信息处理器规则抽取演示 ===")

    # 创建工作流实例
    workflow = RoomMatchingWorkflow()

    # 步骤1参数 - 使用酒店ID数组
    hotel_ids = [
        "2146988",   # 可以配置不同的酒店ID
        "799311",    # 可以配置不同的酒店ID
        "8789588",   # 可以配置不同的酒店ID
        "15896204",  # 可以配置不同的酒店ID
        "2424723",   # 可以配置不同的酒店ID
        "8527690"    # 可以配置不同的酒店ID
    ]
    step1_params = {
        "hotel_ids": hotel_ids,
        "method": "GET"
    }

    try:
        logger.info(f"执行步骤1: HTTP信息处理和规则抽取（批量处理）")
        logger.info(f"处理酒店ID列表: {hotel_ids}")

        # 使用工作流执行批量处理
        result = workflow.execute_batch_step(1, **step1_params)

        if result.get('status') == 'success':
            logger.info("✅ 批量处理执行成功")

            # 输出详细结果
            print("\n" + "="*80)
            print("HTTP信息处理器批量执行结果")
            print("="*80)

            # 批量处理统计
            batch_summary = result.get('batch_summary', {})
            print(f"\n📊 批量处理统计:")
            print(f"   处理酒店数量: {batch_summary.get('total_hotels', 0)}个")
            print(f"   成功处理: {batch_summary.get('successful_hotels', 0)}个")
            print(f"   失败处理: {batch_summary.get('failed_hotels', 0)}个")
            print(f"   成功率: {batch_summary.get('success_rate', '0%')}")

            # 显示每个酒店的处理结果摘要
            hotel_results = result.get('hotel_results', [])
            for hotel_result in hotel_results:
                hotel_id = hotel_result.get('hotel_id', 'Unknown')
                status = hotel_result.get('status', 'unknown')

                if status == 'success':
                    summary = hotel_result.get('room_data_summary', {})
                    hotel_info = summary.get('hotel_info', {})

                    print(f"\n🏨 酒店 {hotel_id} - 处理成功:")
                    print(f"   酒店名称: {hotel_info.get('hotel_name', 'N/A')}")
                    print(f"   英文名称: {hotel_info.get('hotel_name_en', 'N/A')}")
                    print(f"   城市: {hotel_info.get('city', 'N/A')}")
                    print(
                        f"   输入-未匹配房型: {summary.get('input_unmatched_count', 0)}个")
                    print(
                        f"   输入-标准房型: {summary.get('input_standard_count', 0)}个")
                    print(
                        f"   输入-已匹配案例: {summary.get('input_matched_pairs_count', 0)}个")
                    print(
                        f"   LLM-成功匹配: {summary.get('llm_successful_matches', 0)}个")
                    print(
                        f"   LLM-无法匹配: {summary.get('llm_unsuccessful_matches', 0)}个")
                    print(
                        f"   LLM-成功率: {summary.get('llm_success_rate', '0%')}")
                else:
                    error_msg = hotel_result.get('error', '未知错误')
                    print(f"\n❌ 酒店 {hotel_id} - 处理失败: {error_msg}")

            # 显示合并的CSV输出信息
            csv_path = result.get('merged_csv_path', '')
            if csv_path:
                print(f"\n📄 所有酒店的匹配结果已合并保存到CSV文件:")
                print(f"   文件路径: {csv_path}")

                # 检查文件是否存在并显示文件大小
                if os.path.exists(csv_path):
                    file_size = os.path.getsize(csv_path)
                    print(f"   文件大小: {file_size} 字节")

                    # 尝试读取并显示前几行
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_path)
                        print(f"   总记录数量: {len(df)} 条")

                        # 按酒店ID统计
                        if '酒店ID' in df.columns:
                            hotel_counts = df['酒店ID'].value_counts()
                            print(f"   各酒店记录数:")
                            for hotel_id, count in hotel_counts.items():
                                print(f"     酒店 {hotel_id}: {count} 条")

                        if len(df) > 0:
                            print(f"\n📋 合并CSV文件内容预览（前10行）:")
                            print(df.head(10).to_string(index=False))
                    except Exception as e:
                        print(f"   读取CSV文件时出错: {str(e)}")
                else:
                    print(f"   ⚠️  警告: 合并CSV文件不存在")
            else:
                print(f"\n⚠️  警告: 未生成合并CSV文件")

            print(f"\n✅ HTTP信息处理器批量测试完成")
            return result

        else:
            error_msg = result.get('error', '未知错误')
            logger.error(f"❌ 批量处理执行失败: {error_msg}")
            print(f"\n❌ 批量测试失败: {error_msg}")
            return result

    except Exception as e:
        logger.error(f"❌ 执行异常: {str(e)}")
        print(f"\n❌ 执行异常: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    print("房型匹配工作流 - HTTP信息处理器测试")
    print("="*80)

    # 运行HTTP信息处理器演示
    result = demo_http_info_processor()

    # 输出最终状态
    if result.get('status') == 'success':
        print(f"\n🎉 测试完成 - 成功抽取房型匹配规则!")
    else:
        print(f"\n❌ 测试失败!")
