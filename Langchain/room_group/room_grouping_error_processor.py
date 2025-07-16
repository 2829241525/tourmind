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
- **对于餐食（早餐，晚餐等），售后政策（退款等）可以忽略差异**
- **在床型有描述且一致，或床型未描述的情况下，人数描述的差异可以忽略**

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
        """专门处理CSV文件中llm_grouping_decision为NO的数据进行重新审核 - 块写入模式"""
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

            # 重置索引以确保连续性
            filtered_df = filtered_df.reset_index(drop=True)
            logger.info("已重置DataFrame索引以确保数据一致性")

            # 添加新的结果列
            filtered_df['llm_review_decision'] = ''
            if self.include_analysis:
                filtered_df['llm_review_analysis'] = ''

            # 准备输出文件
            output_file = csv_file_path.replace(
                '.csv', '_no_review_results.csv')

            # 写入CSV头部（如果文件不存在）
            if not os.path.exists(output_file):
                # 写入空的DataFrame头部
                empty_df = filtered_df.iloc[0:0].copy()  # 只有列名，没有数据
                empty_df.to_csv(output_file, index=False, encoding='utf-8')
                logger.info(f"创建输出文件并写入CSV头部: {output_file}")

            # 多线程块处理
            total_count = len(filtered_df)
            batch_count = (total_count + batch_size - 1) // batch_size
            logger.info(
                f"开始多线程块处理模式，总数据: {total_count}条，块大小: {batch_size}，块数量: {batch_count}，最大线程数: {max_workers}")

            # 准备块任务
            chunk_tasks = []
            for chunk_start in range(0, total_count, batch_size):
                chunk_end = min(chunk_start + batch_size, total_count)

                # 提取当前块的完整数据
                chunk_df = filtered_df.iloc[chunk_start:chunk_end].copy()
                chunk_num = chunk_start // batch_size + 1

                chunk_tasks.append((chunk_df, chunk_num))

            # 使用线程池处理块，并使用文件锁确保写入安全
            file_lock = threading.Lock()
            processed_chunks = 0
            total_stats = {'yes': 0, 'no': 0, 'error': 0, 'unknown': 0}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有块任务
                future_to_chunk = {
                    executor.submit(self.process_and_write_chunk, chunk_df, chunk_num, output_file, file_lock): chunk_num
                    for chunk_df, chunk_num in chunk_tasks
                }

                # 处理完成的块
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_num = future_to_chunk[future]
                        chunk_stats = future.result()

                        # 累计统计信息
                        for key in total_stats:
                            total_stats[key] += chunk_stats.get(key, 0)

                        processed_chunks += 1
                        progress = (processed_chunks / batch_count) * 100
                        logger.info(
                            f"块 {chunk_num} 处理并写入完成: {processed_chunks}/{batch_count} ({progress:.1f}%)")

                    except Exception as e:
                        chunk_num = future_to_chunk[future]
                        logger.error(f"块 {chunk_num} 处理失败: {str(e)}")
                        processed_chunks += 1

            logger.info(f"所有块处理完成，结果已分块写入: {output_file}")

            # 统计最终结果
            logger.info("重新审核结果统计:")
            logger.info(f"  原始NO数据总数: {total_count}")
            logger.info(f"  重新审核后YES (改为可以分组): {total_stats['yes']}")
            logger.info(f"  重新审核后NO (仍然不可分组): {total_stats['no']}")
            logger.info(f"  ERROR (处理错误): {total_stats['error']}")
            logger.info(f"  UNKNOWN (未知): {total_stats['unknown']}")
            if total_count > 0:
                logger.info(
                    f"  误判率: {total_stats['yes']/total_count*100:.1f}%")

            return output_file

        except Exception as e:
            logger.error(f"处理NO判断重新审核失败: {str(e)}")
            raise

    def process_and_write_chunk(self, chunk_df: pd.DataFrame, chunk_num: int, output_file: str, file_lock: threading.Lock) -> Dict[str, int]:
        """处理一个数据块并立即写入CSV文件"""
        try:
            logger.info(f"开始处理块 {chunk_num}，包含 {len(chunk_df)} 条数据")

            # 准备房型对数据
            room_pairs = []
            for _, row in chunk_df.iterrows():
                spl_room_text = str(row['spl_room_text'])
                s_room_text = str(row['s_room_text'])
                room_pairs.append((spl_room_text, s_room_text))

            # 调用LLM处理这个块
            batch_decisions, batch_analyses = self.call_llm_for_no_review_batch(
                room_pairs)

            # 验证结果数量匹配
            if len(batch_decisions) != len(chunk_df):
                logger.warning(
                    f"块 {chunk_num} 结果数量不匹配: 期望{len(chunk_df)}，实际{len(batch_decisions)}")
                # 补齐缺失的结果
                while len(batch_decisions) < len(chunk_df):
                    batch_decisions.append('ERROR')
                    batch_analyses.append('')

            # 更新chunk数据框
            chunk_df_copy = chunk_df.copy()
            for i, (decision, analysis) in enumerate(zip(batch_decisions, batch_analyses)):
                if i < len(chunk_df_copy):
                    chunk_df_copy.iloc[i, chunk_df_copy.columns.get_loc(
                        'llm_review_decision')] = decision
                    if self.include_analysis:
                        chunk_df_copy.iloc[i, chunk_df_copy.columns.get_loc(
                            'llm_review_analysis')] = analysis

            # 使用文件锁安全写入CSV文件
            with file_lock:
                # 追加写入，不包含头部
                chunk_df_copy.to_csv(
                    output_file, mode='a', header=False, index=False, encoding='utf-8')
                logger.info(f"块 {chunk_num} 已安全写入CSV文件")

            # 统计本块结果
            stats = {
                'yes': chunk_df_copy['llm_review_decision'].value_counts().get('YES', 0),
                'no': chunk_df_copy['llm_review_decision'].value_counts().get('NO', 0),
                'error': chunk_df_copy['llm_review_decision'].value_counts().get('ERROR', 0),
                'unknown': chunk_df_copy['llm_review_decision'].value_counts().get('UNKNOWN', 0)
            }

            logger.info(
                f"块 {chunk_num} 处理完成 - YES: {stats['yes']}, NO: {stats['no']}, ERROR: {stats['error']}, UNKNOWN: {stats['unknown']}")
            return stats

        except Exception as e:
            logger.error(f"处理块 {chunk_num} 失败: {str(e)}")
            # 返回错误统计
            error_stats = {
                'yes': 0,
                'no': 0,
                'error': len(chunk_df),
                'unknown': 0
            }

            # 尝试写入错误结果
            try:
                chunk_df_copy = chunk_df.copy()
                chunk_df_copy['llm_review_decision'] = 'ERROR'
                if self.include_analysis:
                    chunk_df_copy['llm_review_analysis'] = f'处理失败: {str(e)}'

                with file_lock:
                    chunk_df_copy.to_csv(
                        output_file, mode='a', header=False, index=False, encoding='utf-8')
                    logger.info(f"块 {chunk_num} 错误结果已写入CSV文件")
            except Exception as write_error:
                logger.error(f"写入块 {chunk_num} 错误结果失败: {str(write_error)}")

            return error_stats

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

        # 读取分块写入的最终结果进行统计显示
        print("\n=== 读取分块写入的最终结果 ===")
        final_df = pd.read_csv(output_file)
        print(f"最终结果文件包含 {len(final_df)} 条记录")

        # 显示结果示例
        print("\n=== 重新审核结果示例 ===")
        for i, row in final_df.head(10).iterrows():
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
        changed_to_yes = len(
            final_df[final_df['llm_review_decision'] == 'YES'])
        still_no = len(final_df[final_df['llm_review_decision'] == 'NO'])
        error_count = len(final_df[final_df['llm_review_decision'] == 'ERROR'])
        unknown_count = len(
            final_df[final_df['llm_review_decision'] == 'UNKNOWN'])
        total_reviewed = len(final_df)

        print(f"\n=== 最终审核统计结果 ===")
        print(f"总共重新审核的NO数据: {total_reviewed} 条")
        print(
            f"改判为YES (可以分组): {changed_to_yes} 条 ({changed_to_yes/total_reviewed*100:.1f}%)")
        print(
            f"仍然为NO (不可分组): {still_no} 条 ({still_no/total_reviewed*100:.1f}%)")
        print(
            f"处理错误 (ERROR): {error_count} 条 ({error_count/total_reviewed*100:.1f}%)")
        print(
            f"未知状态 (UNKNOWN): {unknown_count} 条 ({unknown_count/total_reviewed*100:.1f}%)")
        if total_reviewed > 0:
            print(f"可能的误判率: {changed_to_yes/total_reviewed*100:.1f}%")
            print(
                f"成功处理率: {(changed_to_yes + still_no)/total_reviewed*100:.1f}%")

    except Exception as e:
        print(f"重新审核失败: {str(e)}")


if __name__ == "__main__":
    main()
