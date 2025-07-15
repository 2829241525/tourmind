#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房间分组错误处理器 - NO判断重新审核专用版
专门用于重新审核CSV文件中llm_grouping_decision为NO的数据

特点：
- 专门针对NO判断数据的重新审核
- 智能识别床数量描述误判
- 处理房型类型与床数量过度关联问题
- 支持多线程并发处理
- 宽松的重新审核标准

使用方法：
python room_grouping_error_processor.py

输出：
- 包含重新审核结果的CSV文件
- 详细的重新审核分析原因
"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import pandas as pd
import os
import sys
import logging
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 优先使用本地langchain源码
LANGCHAIN_MASTER_PATH = "/home/maxon/disk2/roomMatch/room_match/Langchain/langchain-master"
sys.path.insert(0, os.path.join(LANGCHAIN_MASTER_PATH, "libs", "langchain"))
sys.path.insert(0, os.path.join(LANGCHAIN_MASTER_PATH, "libs", "core"))
sys.path.insert(0, os.path.join(
    LANGCHAIN_MASTER_PATH, "libs", "partners", "openai"))

# LLM配置
LLM_CONFIG = {
    'qwen': {
        'model_name': 'deepseek-r1',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'temperature': 0.1,
        'max_tokens': 500,
        'api_key': os.getenv('DASHSCOPE_API_KEY', 'sk-6d6b7b8eac764d08b90a817610f2f6ce')
    }
}

# 设置日志 - 临时调整为DEBUG级别以便追踪问题
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            'room_grouping_error_review.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RoomGroupingErrorProcessor:
    """房间分组错误处理器 - 专门用于NO判断重新审核"""

    def __init__(self, include_analysis: bool = True):
        """初始化处理器"""
        self.include_analysis = include_analysis
        self.llm_config = LLM_CONFIG['qwen']
        self._setup_llm()
        self._setup_no_review_prompt()

    def _setup_llm(self):
        """设置大模型客户端"""
        os.environ['OPENAI_API_KEY'] = self.llm_config['api_key']

        self.llm = ChatOpenAI(
            model=self.llm_config['model_name'],
            temperature=self.llm_config['temperature'],
            max_tokens=self.llm_config['max_tokens'],
            api_key=self.llm_config['api_key'],
            base_url=self.llm_config['base_url']
        )
        logger.info(f"大模型初始化成功: {self.llm_config['model_name']}")

    def _setup_no_review_prompt(self):
        """设置专门用于重新审核NO判断的提示词模板"""
        if self.include_analysis:
            self.no_review_prompt_template = """你现在是一位专业的酒店房型审核员，需要重新审核之前判断为"NO"的房型组合。

重要：之前的判断可能存在以下几种误判情况，请特别关注：

**重点审核项目：**

1. **床数量描述误判**：
   - 检查是否将"1 double bed,1 double bed"这样的重复描述误判为两张床
   - 实际上"1 double bed,1 double bed"应该理解为一张双人床，而不是两张床
   - 类似的重复描述都应该去重处理

2. **房型类型与床数量过度关联**：
   - 房间等级/类型（如standard, superior, deluxe）与床的数量不应该强关联
   - 例如："standard room with 1 double bed" vs "standard room" 应该被视为可以分组
   - 床的配置通常是可以调整的，不应成为房型分组的主要障碍

3. **Standard房型的特殊处理**：
   - "standard"前缀通常可以忽略
   - "standard twin room" 和 "twin room" 应该视为同一等级
   - "standard suite" 和 "suite" 应该视为同一等级

   
**重新审核原则：**

- **消除重复描述**：任何形式的重复都要去重（如"2 double beds,2 double beds"视为"2 double beds"）
- **关注房型等级一致性**：相同等级的房型（如都是standard、都是superior）应该优先考虑分组
- **床数量宽松处理**：在房型等级一致的前提下，床数量差异不应该成为拒绝分组的主要原因
- **床大小和房间大小宽松处理**：客户不能够明显感知到床大小和空间大小，只要床大体在同一等级，床大小和房间大小差异不应该成为拒绝分组的主要原因。详细重审因空间大小和床大小差异而拒绝分组的原因。
- **人数匹配优先**：如果两个房型都能满足相同的入住人数要求，应该优先考虑分组

**针对常见误判的修正：**
- 如果之前因为"床数量不匹配"而判断为NO，请重新检查是否存在重复描述
- 如果之前因为"房型降级"而判断为NO，请重新检查是否房型等级实际上是一致的
- 如果之前因为过于严格的床型要求而判断为NO，请以更宽松的标准重新评估

房型组合数据：
{room_pairs}

**重要：必须严格按照输入顺序返回结果，不能跳过任何组号！**

请按照以下格式返回结果，每组一行，必须包含组号：
组1: YES | 重新审核后认为可以分组，原因：xxx
组2: NO  | 仍然认为不可分组，原因：xxx（必须是客户能明显感知的实质性差异）
组3: YES | 重新审核后认为可以分组，原因：xxx
...

**注意：**
- 必须按照组1、组2、组3...的顺序返回
- 每行必须以"组X:"开头，其中X是组号
- 不能跳过任何组号
- 返回的组数必须与输入的组数完全一致

**审核要求：**
- 对于每个之前判断为NO的组合，请用更宽松和实用的标准重新评估
- 只有当存在客户能够明显感知且会实际投诉的差异时，才维持NO判断
- 优先考虑房型等级一致性和入住人数匹配
- 床型差异、描述详细程度差异等不应成为NO的主要原因
- 如果重新判断为YES，请说明为什么之前的判断可能是误判
- 如果维持NO，请明确指出客户会投诉的具体实质性差异
"""

    def _parse_batch_results(self, llm_response: str, expected_count: int) -> Tuple[List[str], List[str]]:
        """解析批量LLM返回结果，严格按组号顺序匹配，返回(判断结果列表, 分析原因列表)"""
        try:
            # 用字典存储按组号索引的结果
            group_results = {}
            lines = llm_response.strip().split('\n')

            logger.debug(f"原始LLM响应:\n{llm_response}")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 检查是否以"组X:"开头
                if line.startswith('组') and ':' in line:
                    try:
                        # 提取组号
                        group_part = line.split(':', 1)[0]
                        group_num_str = group_part.replace('组', '').strip()
                        group_num = int(group_num_str) - 1  # 转换为0基索引

                        if group_num < 0 or group_num >= expected_count:
                            logger.warning(
                                f"组号超出范围: {group_num}, 期望范围: 0-{expected_count-1}")
                            continue

                        # 提取决策和分析
                        content = line.split(':', 1)[1].strip()

                        if self.include_analysis and '|' in content:
                            parts = content.split('|', 1)
                            judgment = parts[0].strip().upper()
                            analysis = parts[1].strip() if len(
                                parts) > 1 else ""
                        else:
                            judgment = content.upper()
                            analysis = ""

                        # 判断决策
                        if 'YES' in judgment:
                            decision = 'YES'
                        elif 'NO' in judgment:
                            decision = 'NO'
                        else:
                            decision = 'UNKNOWN'

                        group_results[group_num] = (decision, analysis)
                        logger.debug(
                            f"解析组{group_num + 1}: {decision} | {analysis[:50]}...")

                    except (ValueError, IndexError) as e:
                        logger.warning(f"解析组号失败: {line}, 错误: {e}")
                        continue

            # 按顺序构建结果列表
            decisions = []
            analyses = []

            for i in range(expected_count):
                if i in group_results:
                    decision, analysis = group_results[i]
                    decisions.append(decision)
                    analyses.append(analysis)
                    logger.debug(f"组{i+1}已匹配: {decision}")
                else:
                    decisions.append('UNKNOWN')
                    analyses.append("")
                    logger.warning(f"组{i+1}缺失结果，设置为UNKNOWN")

            logger.info(
                f"严格解析完成，预期{expected_count}个结果，实际解析{len(group_results)}个有效组，最终输出{len(decisions)}个结果")

            # 验证结果
            if len(decisions) != expected_count:
                logger.error(f"结果数量不匹配! 预期{expected_count}，实际{len(decisions)}")

            return decisions, analyses

        except Exception as e:
            logger.error(f"批量结果解析失败: {str(e)}")
            return ['ERROR'] * expected_count, [''] * expected_count

    def process_no_decisions_csv(self, csv_file_path: str, max_rows: int = None,
                                 batch_size: int = 20, max_workers: int = 20) -> str:
        """专门处理CSV文件中llm_grouping_decision为NO的数据进行重新审核"""
        try:
            # 读取CSV文件
            logger.info(f"开始读取CSV文件进行NO判断重新审核: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            logger.info(f"CSV文件读取成功，总行数: {len(df)}")

            # 筛选llm_grouping_decision为NO的数据
            filtered_df = df[df['llm_grouping_decision'] == 'NO'].copy()
            logger.info(
                f"筛选出llm_grouping_decision为NO的数据: {len(filtered_df)} 行")

            if len(filtered_df) == 0:
                logger.info("没有需要重新审核的NO判断数据")
                return csv_file_path

            # 限制处理量
            if max_rows and len(filtered_df) > max_rows:
                filtered_df = filtered_df.head(max_rows)
                logger.info(f"限制处理数量为: {max_rows} 行")

            # 重置索引以确保连续性，避免多线程索引错位
            filtered_df = filtered_df.reset_index(drop=True)
            logger.info("已重置DataFrame索引以确保数据一致性")

            # 添加新的结果列
            filtered_df['llm_review_decision'] = ''
            if self.include_analysis:
                filtered_df['llm_review_analysis'] = ''

            # 多线程批量处理
            total_count = len(filtered_df)
            batch_count = (total_count + batch_size - 1) // batch_size  # 向上取整
            logger.info(
                f"开始多线程批量重新审核，总数据: {total_count}条，批次大小: {batch_size}，批次数量: {batch_count}，最大线程数: {max_workers}")

            # 准备批次任务
            batch_tasks = []
            for batch_start in range(0, total_count, batch_size):
                batch_end = min(batch_start + batch_size, total_count)

                # 使用iloc确保连续索引访问
                room_pairs = []
                batch_indices = []
                for i in range(batch_start, batch_end):
                    row = filtered_df.iloc[i]
                    spl_room_text = str(row['spl_room_text'])
                    s_room_text = str(row['s_room_text'])
                    room_pairs.append((spl_room_text, s_room_text))
                    batch_indices.append(i)  # 使用连续的索引

                batch_tasks.append(
                    (room_pairs, batch_indices, batch_start // batch_size + 1))

            # 使用线程池处理批次
            results = {}  # 存储结果，key为index
            processed_batches = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有批次任务，使用重新审核的方法
                future_to_batch = {
                    executor.submit(self.call_llm_for_no_review_batch, room_pairs): (batch_indices, batch_num)
                    for room_pairs, batch_indices, batch_num in batch_tasks
                }

                # 处理完成的批次
                for future in as_completed(future_to_batch):
                    try:
                        batch_indices, batch_num = future_to_batch[future]
                        batch_decisions, batch_analyses = future.result()

                        # 存储批次结果 - 确保数据对应正确
                        if len(batch_decisions) != len(batch_indices):
                            logger.warning(
                                f"批次 {batch_num} 返回结果数量不匹配: 期望{len(batch_indices)}，实际{len(batch_decisions)}")

                        # 详细记录每个结果的映射关系
                        logger.info(f"批次 {batch_num} 结果映射:")
                        for i, (idx, decision, analysis) in enumerate(zip(batch_indices, batch_decisions, batch_analyses)):
                            # 获取对应的房型数据用于验证
                            if idx < len(filtered_df):
                                row = filtered_df.iloc[idx]
                                spl_text = str(row['spl_room_text'])[:40]
                                s_text = str(row['s_room_text'])[:40]
                                logger.info(
                                    f"  索引{idx}(组{i+1}): '{spl_text}...' vs '{s_text}...' -> {decision}")

                            results[idx] = (decision, analysis)

                        processed_batches += 1
                        processed_count = processed_batches * batch_size
                        if processed_count > total_count:
                            processed_count = total_count

                        logger.info(
                            f"重新审核批次 {batch_num} 处理完成: {processed_batches}/{batch_count} ({processed_batches/batch_count*100:.1f}%)")

                    except Exception as e:
                        batch_indices, batch_num = future_to_batch[future]
                        logger.error(f"重新审核批次 {batch_num} 处理失败: {str(e)}")
                        # 对失败的批次中的所有索引设置ERROR
                        for idx in batch_indices:
                            results[idx] = ('ERROR', '')
                        processed_batches += 1

            # 按照连续索引顺序更新DataFrame
            logger.info("按连续索引顺序更新重新审核结果...")
            for i in range(len(filtered_df)):
                if i in results:
                    decision, analysis = results[i]
                    filtered_df.at[i, 'llm_review_decision'] = decision
                    if self.include_analysis:
                        filtered_df.at[i, 'llm_review_analysis'] = analysis
                else:
                    # 如果某个索引没有结果，设置为ERROR
                    filtered_df.at[i, 'llm_review_decision'] = 'ERROR'
                    if self.include_analysis:
                        filtered_df.at[i, 'llm_review_analysis'] = ''

            # 保存结果
            output_file = csv_file_path.replace(
                '.csv', '_no_review_results.csv')
            filtered_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"NO判断重新审核完成，结果已保存到: {output_file}")

            # 统计结果
            yes_count = len(
                filtered_df[filtered_df['llm_review_decision'] == 'YES'])
            no_count = len(
                filtered_df[filtered_df['llm_review_decision'] == 'NO'])
            error_count = len(
                filtered_df[filtered_df['llm_review_decision'] == 'ERROR'])
            unknown_count = len(
                filtered_df[filtered_df['llm_review_decision'] == 'UNKNOWN'])

            logger.info("重新审核结果统计:")
            logger.info(f"  原始NO数据总数: {len(filtered_df)}")
            logger.info(f"  重新审核后YES (改为可以分组): {yes_count}")
            logger.info(f"  重新审核后NO (仍然不可分组): {no_count}")
            logger.info(f"  ERROR (处理错误): {error_count}")
            logger.info(f"  UNKNOWN (未知): {unknown_count}")
            logger.info(f"  误判率: {yes_count/len(filtered_df)*100:.1f}%")

            return output_file

        except Exception as e:
            logger.error(f"处理NO判断重新审核失败: {str(e)}")
            raise

    def call_llm_for_no_review_batch(self, room_pairs: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        """专门用于重新审核NO判断的批量LLM调用"""
        try:
            # 构建输入内容并记录调试信息
            room_pairs_text = ""
            logger.debug(f"批次包含{len(room_pairs)}组房型对:")
            for i, (spl_room_text, s_room_text) in enumerate(room_pairs, 1):
                room_pairs_text += f"组{i}:\n"
                room_pairs_text += f"  房型A（供应商房型）: {spl_room_text}\n"
                room_pairs_text += f"  房型B（标准房型）: {s_room_text}\n\n"
                logger.debug(f"  组{i}: '{spl_room_text}' vs '{s_room_text}'")

            prompt = self.no_review_prompt_template.format(
                room_pairs=room_pairs_text)

            # 记录发送给LLM的完整输入
            logger.debug("发送给LLM的输入房型对:\n" + room_pairs_text.strip())

            messages = [HumanMessage(content=prompt)]
            result = self.llm.invoke(messages)

            llm_response = result.content if hasattr(
                result, 'content') else str(result)

            # 记录LLM的原始响应
            logger.debug(f"LLM原始响应:\n{llm_response}")

            decisions, analyses = self._parse_batch_results(
                llm_response, len(room_pairs))

            # 验证输入输出对应关系
            logger.info(f"批次处理完成: 输入{len(room_pairs)}组，输出{len(decisions)}个决策")
            for i, (decision, analysis) in enumerate(zip(decisions, analyses)):
                if i < len(room_pairs):
                    spl_text, s_text = room_pairs[i]
                    logger.debug(
                        f"组{i+1}映射: '{spl_text[:30]}...' vs '{s_text[:30]}...' -> {decision}")

            return decisions, analyses

        except Exception as e:
            logger.error(f"NO判断重新审核批量调用LLM失败: {str(e)}")
            return ['ERROR'] * len(room_pairs), [''] * len(room_pairs)


def main():
    """主函数"""
    # 指定用户提供的CSV文件路径
    csv_file_path = "/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/room_group_train_fix_llm_processed.csv"

    # 创建处理器（默认包含分析）
    processor = RoomGroupingErrorProcessor(include_analysis=True)

    print("开始重新审核CSV文件中所有llm_grouping_decision为NO的数据...")
    print(f"处理文件: {csv_file_path}")

    try:
        # 使用NO判断重新审核方法
        output_file = processor.process_no_decisions_csv(
            csv_file_path,
            max_rows=2000,  # 处理所有NO数据，不限制数量
            batch_size=20,
            max_workers=10
        )

        print(f"重新审核完成，输出文件: {output_file}")

        # 显示结果示例
        print("\n=== 重新审核结果示例 ===")
        df = pd.read_csv(output_file)

        # 显示原始NO数据的重新审核结果
        for i, row in df.head(10).iterrows():
            print(f"数据{i+1}:")
            print(f"  房型A: {row['spl_room_text']}")
            print(f"  房型B: {row['s_room_text']}")
            print(f"  原始判断: {row['llm_grouping_decision']}")
            print(f"  原始分析: {row['llm_grouping_analysis']}")
            print(f"  重新审核判断: {row['llm_review_decision']}")
            print(f"  重新审核分析: {row['llm_review_analysis']}")
            print(
                f"  判断是否改变: {'是' if row['llm_grouping_decision'] != row['llm_review_decision'] else '否'}")
            print("-" * 80)

        # 统计改变的情况
        changed_to_yes = len(df[df['llm_review_decision'] == 'YES'])
        still_no = len(df[df['llm_review_decision'] == 'NO'])
        total_reviewed = len(df)

        print(f"\n=== 审核统计结果 ===")
        print(f"总共重新审核的NO数据: {total_reviewed} 条")
        print(
            f"改判为YES (可以分组): {changed_to_yes} 条 ({changed_to_yes/total_reviewed*100:.1f}%)")
        print(
            f"仍然为NO (不可分组): {still_no} 条 ({still_no/total_reviewed*100:.1f}%)")
        print(f"可能的误判率: {changed_to_yes/total_reviewed*100:.1f}%")

    except Exception as e:
        print(f"重新审核失败: {str(e)}")


if __name__ == "__main__":
    main()
