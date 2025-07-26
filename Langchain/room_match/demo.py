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

    # 步骤1参数 - 使用酒店ID
    hotel_id = "2146988"  # 可以配置不同的酒店ID
    # hotel_id = "799311"  # 可以配置不同的酒店ID
    # hotel_id = "8789588"  # 可以配置不同的酒店ID
    # hotel_id = "15896204"  # 可以配置不同的酒店ID
    # hotel_id = "2424723"  # 可以配置不同的酒店ID
    # hotel_id = "8527690"  # 可以配置不同的酒店ID
    step1_params = {
        "hotel_id": hotel_id,
        "method": "GET"
    }

    try:
        logger.info(f"执行步骤1: HTTP信息处理和规则抽取")
        logger.info(f"使用酒店ID: {hotel_id}")

        # 使用工作流执行单个步骤
        result = workflow.execute_single_step(1, **step1_params)

        if result.get('status') == 'success':
            logger.info("✅ 步骤1执行成功")

            # 输出详细结果
            print("\n" + "="*80)
            print("HTTP信息处理器执行结果")
            print("="*80)

            # 基本信息
            summary = result.get('room_data_summary', {})
            hotel_info = summary.get('hotel_info', {})

            print(f"\n🏨 酒店信息:")
            print(f"   酒店ID: {hotel_info.get('hotel_id', 'N/A')}")
            print(f"   酒店名称: {hotel_info.get('hotel_name', 'N/A')}")
            print(f"   英文名称: {hotel_info.get('hotel_name_en', 'N/A')}")
            print(f"   城市: {hotel_info.get('city', 'N/A')}")

            print(f"\n📊 房型数据统计:")
            print(f"   未匹配供应商房型: {summary.get('unmatched_count', 0)}个")
            print(f"   标准房型: {summary.get('standard_count', 0)}个")
            print(f"   已匹配案例: {summary.get('matched_count', 0)}个")

            # 显示格式化的房型数据
            formatted_data = summary.get('formatted_data', {})
            if formatted_data:
                print(f"\n📝 格式化房型数据:")
                print("="*60)

                matched_examples = formatted_data.get('matched_examples', [])
                if matched_examples:
                    print("已匹配案例参考:")
                    for example in matched_examples:
                        print(f"  {example}")
                    print()

                unmatched_rooms = formatted_data.get('unmatched_rooms', [])
                if unmatched_rooms:
                    print("待匹配的供应商房型:")
                    for room in unmatched_rooms:
                        print(f"  {room}")
                    print()

                standard_rooms = formatted_data.get('standard_rooms', [])
                if standard_rooms:
                    print("可选择的标准房型:")
                    for room in standard_rooms:
                        print(f"  {room}")

                print("="*60)

            # 显示格式化的prompt
            formatted_prompt = result.get('formatted_prompt', '')
            if formatted_prompt:
                print(f"\n📝 转换后的完整Prompt:")
                print("="*60)
                print(formatted_prompt)
                print("="*60)

            # 显示大模型生成的规则
            elapsed_time = result.get('elapsed_time', 0)
            print(f"\n🔧 大模型生成的匹配规则 (耗时: {elapsed_time:.2f}秒):")
            print("="*60)
            print(result.get('llm_response', ''))
            print(result.get('agent_query', ''))
            print(result.get('agent_response', ''))
            print("="*60)

            # 显示CSV输出信息
            csv_output = result.get('csv_output', {})
            if csv_output:
                print(f"\n📄 CSV输出结果:")
                print("="*60)
                csv_filepath = csv_output.get('filepath', '')
                parsed_matches_count = csv_output.get(
                    'parsed_matches_count', 0)
                parsed_matches = csv_output.get('parsed_matches', [])

                if csv_filepath:
                    print(f"   CSV文件路径: {csv_filepath}")
                    print(f"   解析的匹配关系数量: {parsed_matches_count}")

                    if parsed_matches:
                        print(f"\n   解析出的匹配关系:")
                        # 只显示前10个
                        for i, match in enumerate(parsed_matches[:10], 1):
                            supplier_room = match.get('supplier_room', '')
                            standard_room = match.get('standard_room', '')
                            print(
                                f"     [{i}] {supplier_room} → {standard_room}")

                        if len(parsed_matches) > 10:
                            print(
                                f"     ... 还有{len(parsed_matches) - 10}个匹配关系")

                    # 尝试读取并显示CSV文件的前几行
                    try:
                        import csv
                        print(f"\n   CSV文件内容预览:")
                        with open(csv_filepath, 'r', encoding='utf-8') as f:
                            csv_reader = csv.reader(f)
                            rows = list(csv_reader)

                            # 显示表头
                            if rows:
                                headers = rows[0]
                                print(f"     表头: {' | '.join(headers)}")

                                # 显示前5行数据
                                data_rows = rows[1:6] if len(rows) > 1 else []
                                for i, row in enumerate(data_rows, 1):
                                    print(f"     数据[{i}]: {' | '.join(row)}")

                                if len(rows) > 6:
                                    print(f"     ... CSV文件共包含{len(rows)-1}行数据")

                    except Exception as e:
                        print(f"     读取CSV文件时出错: {str(e)}")

                else:
                    print("   未生成CSV文件")

                print("="*60)

            print(f"\n✅ HTTP信息处理器测试完成")
            return result

        else:
            error_msg = result.get('error', '未知错误')
            logger.error(f"❌ 步骤1执行失败: {error_msg}")
            print(f"\n❌ 测试失败: {error_msg}")
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
