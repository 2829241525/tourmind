#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试NO判断重新审核功能
"""

import pandas as pd
from room_grouping_error_processor import RoomGroupingErrorProcessor
import sys
import os
sys.path.append('/home/maxon/disk2/roomMatch/room_match/Langchain/room_group')


def test_no_review():
    """测试NO判断重新审核功能"""

    # 指定CSV文件路径
    csv_file_path = "/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/room_group_train_fix_clean_filtered_set_different_with_similarity_with_grouping.csv"

    print("=== 测试NO判断重新审核功能 ===")
    print(f"处理文件: {csv_file_path}")

    # 首先检查文件是否存在以及NO数据的数量
    try:
        df = pd.read_csv(csv_file_path)
        print(f"CSV文件总行数: {len(df)}")

        no_decisions = df[df['llm_grouping_decision'] == 'NO']
        print(f"llm_grouping_decision为NO的数据: {len(no_decisions)} 行")

        if len(no_decisions) == 0:
            print("没有NO判断数据需要重新审核")
            return

        # 显示前几个NO判断的示例
        print("\n=== NO判断数据示例 ===")
        for i, row in no_decisions.head(3).iterrows():
            print(f"示例{i+1}:")
            print(f"  房型A: {row['spl_room_text']}")
            print(f"  房型B: {row['s_room_text']}")
            print(f"  原始分析: {row['llm_grouping_analysis']}")
            print()

        # 创建处理器
        processor = RoomGroupingErrorProcessor(include_analysis=True)

        # 测试处理少量数据（前10条NO数据）
        print("开始测试重新审核功能（处理前10条NO数据）...")
        output_file = processor.process_no_decisions_csv(
            csv_file_path,
            max_rows=10,  # 只处理前10条进行测试
            batch_size=5,
            max_workers=2
        )

        print(f"测试完成，输出文件: {output_file}")

        # 显示测试结果
        result_df = pd.read_csv(output_file)
        print(f"\n=== 测试结果 ===")
        for i, row in result_df.iterrows():
            print(f"测试数据{i+1}:")
            print(f"  房型A: {row['spl_room_text']}")
            print(f"  房型B: {row['s_room_text']}")
            print(f"  原始判断: {row['llm_grouping_decision']}")
            print(f"  重新审核判断: {row['llm_review_decision']}")
            print(f"  重新审核分析: {row['llm_review_analysis']}")
            print(
                f"  判断是否改变: {'是' if row['llm_grouping_decision'] != row['llm_review_decision'] else '否'}")
            print("-" * 60)

        # 统计
        changed_count = len(
            result_df[result_df['llm_review_decision'] == 'YES'])
        total_count = len(result_df)
        print(f"\n测试统计: {changed_count}/{total_count} 条数据改判为YES")

    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_no_review()
