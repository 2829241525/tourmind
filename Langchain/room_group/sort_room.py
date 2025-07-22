#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房间分组处理器 v2 - 优化版
读取CSV文件，获取similarity_score在0.9分以下的数据
调用LLM判断spl_room_text和s_room_text是否能分组

特点：
- 基于真实客户感知能力判断，优先级处理（等级>人数>其他）
- 同等级房型兼容性强化，避免床型细节过度纠结
- 不考虑房间空间匹配问题，专注实质性差异
- 支持多线程并发处理，最大20个线程，大幅提升效率
- 特殊处理笼统描述，避免误判降级风险

使用方法：
python room_grouping_processor_v2.py

输出：
- 包含分组判断结果的CSV文件
- 详细的客户投诉分析原因
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

# 导入langchain组件


# LLM配置qwen-plus DeepSeek-R1-0528
LLM_CONFIG = {
    'qwen': {
        'model_name': 'deepseek-r1',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'temperature': 0.1,
        'max_tokens': 500,
        'api_key': os.getenv('DASHSCOPE_API_KEY', 'sk-6d6b7b8eac764d08b90a817610f2f6ce')
    }
}

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('room_grouping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RoomGroupingProcessor:
    """房间分组处理器"""

    def __init__(self, include_analysis: bool = True):
        """初始化处理器"""
        self.include_analysis = include_analysis
        self.llm_config = LLM_CONFIG['qwen']
        self._setup_llm()
        self._setup_prompt()

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

    def _setup_prompt(self):
        """设置提示词模板"""
        if self.include_analysis:
            self.prompt_template = """你现在是一位普通酒店客户，请从真实客户体验的角度判断以下房型组合是否可以分为同一组。

请对每组房型进行判断，模拟两种入住场景是否会产生客户投诉：
- 场景1: 客户预订时看到房型A，实际入住时提供房型B
- 场景2: 客户预订时看到房型B，实际入住时提供房型A

分析步骤：
1. **首先消除重复描述和冗余信息**：忽略重复词汇（如"2 double beds,2 double beds"视为"2 double beds"）
2. **提取核心信息**：房间等级、床型、床数量、关键设施、景观等
3. **按最低配置理解模糊描述**：对于不明确的描述，按最基础配置处理
4. **评估or选择的风险**：描述中有"or"表示可能提供任意一种，需考虑最差情况

判断原则（基于客户真实感知能力，按优先级排序）：

**优先级1：房型等级匹配**
- 同等级房型（如superior vs superior，classic vs classic）通常可以分组
- 不同等级房型（如standard vs deluxe vs suite）客户能明显感知，会投诉

**优先级2：入住人数匹配**
- 两个房型都能满足标明的入住人数要求时，优先考虑分组
- 例如：都标明"for 1 adults"或都能容纳相同人数

**优先级3：实质性设施差异**
- 关键设施差异：有厨房 vs 无厨房，有阳台 vs 无阳台
- 景观实质差异：海景 vs 山景 vs 无景观

**在房型等级一致且入住人数匹配的前提下，以下差异客户通常不会投诉（可以分组）**：
- 床型描述差异：twin vs double vs 未明确床型（只要能满足入住人数）
- 床数量差异：只要能满足每人有床睡（例如1人入住，1张床和2张床都可接受）
- 人数标注差异：一方明确标注"for X adults"，另一方未标注（同等级房型能容纳相同人数）
- 房间空间担忧：不考虑房间大小或空间匹配问题，同等级房型空间基本一致
- 重复描述差异：如"2 double beds,2 double beds" vs "2 double beds"，因为"2 double beds,2 double beds"可简化为"2 double beds"。类似的场景同样适用。
- 表述顺序差异：如"standard triple room" vs "triple standard room"
- 语言差异：如"双人房" vs "double room"
- 描述详细程度差异：如"superior room" vs "superior twin"（等级一致）
- 床型尺寸微小差别：queen vs full（客户往往无法精确区分）
- 房间面积的小幅差异
- 装修时间等附加信息：newly renovated vs 普通

**笼统描述的特殊处理**：
- 笼统房型描述，根据房型推断默认人数，在设施一致的情况下，人数差异可以忽略。
- 如果两个房型等级相同且都能满足入住人数，笼统描述不视为降级
- 不要因为人数标注差异而担心房间空间或设施不匹配
- 对于任何等级的房型，只要等级一致且能容纳相同人数，一方有详细描述（包含床型、人数标注等），另一方描述笼统，都可以分组


房型组合数据：
{room_pairs}

请按照以下格式返回结果，每组一行，必须包含组号：
组1: NO  | 作为客户我会因为：xxx（具体原因）而进行投诉。如果是床数量问题，请说明A的床数量是多少，B的床数量是多少。
组2: YES | 不投诉
组3: NO  | 作为客户我会因为：xxx（具体原因）而进行投诉
...

**注意：**
- 必须按照组1、组2、组3...的顺序返回
- 每行必须以"组X:"开头，其中X是组号
- 不能跳过任何组号
- 返回的组数必须与输入的组数完全一致

分析要求：
- **按优先级顺序判断**：
  1. 首先检查房型等级是否一致（superior、classic、deluxe等）
  2. 然后确认是否都能满足入住人数要求
  3. 最后考虑其他实质性差异
  4. 对于返回的NO的都要进行回看，是否有重复词汇的判断错误的，如果有，请重新判断；
    是否有未标明人数和床数的，没有默认为1人入住而造成判断错误的，如果有，请重新判断。
    是否有带有standard描述的房型判断错误的，如果有，请根据standard忽略的规则重新判断。
- 所有的standard描述，standard这个词汇是可以忽略的，比如standard twin room 和 twin room 是同一种房型等级，suite 和standard suite是同一种房型等级
- **等级一致且人数匹配的原则**：如果两个房型等级相同且都能满足入住人数，默认判断为YES
- **床型差异的宽容处理**：在等级一致且人数匹配的前提下，床型差异（twin、double、未明确等）不应导致NO判断
- **投诉门槛要求**：只有客户大概率会投诉的明显实质性差异才判断为NO，否则给出YES
- **对于餐食，售后政策可以忽略差异**
- **在床型有描述且一致，或床型未描述的情况下，人数描述的差异可以忽略**
- 如果判断为NO，请明确说明客户能够明显感知到的实质性差异，并且该差异不是床型细节问题
- 如果判断为YES，无需说明详细原因
- 对于重复描述、表述顺序差异、同义词差异等，应判断为YES
- 匹配可以激进一些，优先考虑客户的实际感知而非描述的字面差异
"""

        logger.info("提示词模板设置完成")

    def _build_batch_prompt(self, room_pairs: List[Tuple[str, str]]) -> str:
        """构建批量处理的prompt"""
        room_pairs_text = ""
        for i, (spl_room_text, s_room_text) in enumerate(room_pairs, 1):
            room_pairs_text += f"组{i}:\n"
            room_pairs_text += f"  房型A（供应商房型）: {spl_room_text}\n"
            room_pairs_text += f"  房型B（标准房型）: {s_room_text}\n\n"

        return self.prompt_template.format(room_pairs=room_pairs_text)

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

            # 记录结果分布
            yes_count = decisions.count('YES')
            no_count = decisions.count('NO')
            unknown_count = decisions.count('UNKNOWN')
            error_count = decisions.count('ERROR')
            logger.info(
                f"本批次结果分布 - YES: {yes_count}, NO: {no_count}, UNKNOWN: {unknown_count}, ERROR: {error_count}")

            # 验证结果
            if len(decisions) != expected_count:
                logger.error(f"结果数量不匹配! 预期{expected_count}，实际{len(decisions)}")

            return decisions, analyses

        except Exception as e:
            logger.error(f"批量结果解析失败: {str(e)}")
            logger.error(f"LLM响应内容: {llm_response[:500]}...")
            return ['ERROR'] * expected_count, [''] * expected_count

    def call_llm_for_single_grouping(self, spl_room_text: str, s_room_text: str, index: int) -> Tuple[int, str, str]:
        """单个房间对的LLM调用"""
        try:
            # 构建单个房间对的prompt
            room_pairs_text = f"组1:\n  房型A（供应商房型）: {spl_room_text}\n  房型B（标准房型）: {s_room_text}\n\n"
            prompt = self.prompt_template.format(room_pairs=room_pairs_text)

            messages = [HumanMessage(content=prompt)]
            result = self.llm.invoke(messages)

            llm_response = result.content if hasattr(
                result, 'content') else str(result)
            # 基于组号解析结果
            decisions, analyses = self._parse_batch_results(llm_response, 1)

            return index, decisions[0], analyses[0] if analyses else ""

        except Exception as e:
            logger.error(f"单个LLM调用失败 (index {index}): {str(e)}")
            return index, 'ERROR', ''

    def call_llm_for_batch_grouping(self, room_pairs: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        """批量调用LLM判断房间是否能分组"""
        try:
            prompt = self._build_batch_prompt(room_pairs)
            messages = [HumanMessage(content=prompt)]
            result = self.llm.invoke(messages)

            llm_response = result.content if hasattr(
                result, 'content') else str(result)
            # 基于组号解析结果
            return self._parse_batch_results(llm_response, len(room_pairs))

        except Exception as e:
            logger.error(f"批量调用LLM失败: {str(e)}")
            return ['ERROR'] * len(room_pairs), [''] * len(room_pairs)

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
            batch_decisions, batch_analyses = self.call_llm_for_batch_grouping(
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
                        'llm_grouping_decision')] = decision
                    if self.include_analysis:
                        chunk_df_copy.iloc[i, chunk_df_copy.columns.get_loc(
                            'llm_grouping_analysis')] = analysis

            # 使用文件锁安全写入CSV文件
            with file_lock:
                # 追加写入，不包含头部
                chunk_df_copy.to_csv(
                    output_file, mode='a', header=False, index=False, encoding='utf-8')
                logger.info(f"块 {chunk_num} 已安全写入CSV文件")

            # 统计本块结果
            stats = {
                'yes': chunk_df_copy['llm_grouping_decision'].value_counts().get('YES', 0),
                'no': chunk_df_copy['llm_grouping_decision'].value_counts().get('NO', 0),
                'error': chunk_df_copy['llm_grouping_decision'].value_counts().get('ERROR', 0),
                'unknown': chunk_df_copy['llm_grouping_decision'].value_counts().get('UNKNOWN', 0)
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
                chunk_df_copy['llm_grouping_decision'] = 'ERROR'
                if self.include_analysis:
                    chunk_df_copy['llm_grouping_analysis'] = f'处理失败: {str(e)}'

                with file_lock:
                    chunk_df_copy.to_csv(
                        output_file, mode='a', header=False, index=False, encoding='utf-8')
                    logger.info(f"块 {chunk_num} 错误结果已写入CSV文件")
            except Exception as write_error:
                logger.error(f"写入块 {chunk_num} 错误结果失败: {str(write_error)}")

            return error_stats

    def process_csv_file(self, csv_file_path: str, similarity_threshold: float = 0.9,
                         max_rows: int = 20, batch_size: int = 20, max_workers: int = 20) -> str:
        """处理CSV文件 - 块写入模式"""
        try:
            # 读取CSV文件
            logger.info(f"开始读取CSV文件: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            logger.info(f"CSV文件读取成功，总行数: {len(df)}")

            # 筛选数据
            filtered_df = df[df['similarity_score']
                             < similarity_threshold].copy()
            logger.info(
                f"筛选出similarity_score < {similarity_threshold}的数据: {len(filtered_df)} 行")

            if len(filtered_df) == 0:
                logger.info("没有需要处理的数据")
                return csv_file_path

            # 限制处理量
            if max_rows and len(filtered_df) > max_rows:
                filtered_df = filtered_df.head(max_rows)
                logger.info(f"限制处理数量为: {max_rows} 行")

            # 重置索引以确保连续性
            filtered_df = filtered_df.reset_index(drop=True)
            logger.info("已重置DataFrame索引以确保数据一致性")

            # 添加结果列
            filtered_df['llm_grouping_decision'] = ''
            if self.include_analysis:
                filtered_df['llm_grouping_analysis'] = ''

            # 准备输出文件
            output_file = csv_file_path.replace('.csv', '_with_grouping.csv')

            # 写入CSV头部（如果文件不存在）
            if not os.path.exists(output_file):
                # 写入空的DataFrame头部
                empty_df = filtered_df.iloc[0:0].copy()  # 只有列名，没有数据
                empty_df.to_csv(output_file, index=False, encoding='utf-8')
                logger.info(f"创建输出文件并写入CSV头部: {output_file}")

            # 多线程块处理
            total_count = len(filtered_df)
            chunk_count = (total_count + batch_size - 1) // batch_size
            logger.info(
                f"开始多线程块处理模式，总数据: {total_count}条，块大小: {batch_size}，块数量: {chunk_count}，最大线程数: {max_workers}")

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
                        progress = (processed_chunks / chunk_count) * 100
                        logger.info(
                            f"块 {chunk_num} 处理并写入完成: {processed_chunks}/{chunk_count} ({progress:.1f}%)")

                    except Exception as e:
                        chunk_num = future_to_chunk[future]
                        logger.error(f"块 {chunk_num} 处理失败: {str(e)}")
                        processed_chunks += 1

            logger.info(f"所有块处理完成，结果已分块写入: {output_file}")

            # 统计最终结果
            logger.info("分组结果统计:")
            logger.info(f"  总处理数据: {total_count}")
            logger.info(f"  YES (可以分组): {total_stats['yes']}")
            logger.info(f"  NO (不可以分组): {total_stats['no']}")
            logger.info(f"  ERROR (处理错误): {total_stats['error']}")
            logger.info(f"  UNKNOWN (未知): {total_stats['unknown']}")
            if total_count > 0:
                yes_rate = (total_stats['yes'] / total_count) * 100
                no_rate = (total_stats['no'] / total_count) * 100
                success_rate = (
                    (total_stats['yes'] + total_stats['no']) / total_count) * 100
                logger.info(f"  分组成功率: {yes_rate:.1f}%")
                logger.info(f"  不分组率: {no_rate:.1f}%")
                logger.info(f"  总体处理成功率: {success_rate:.1f}%")

            return output_file

        except Exception as e:
            logger.error(f"处理CSV文件失败: {str(e)}")
            raise


def main():
    """主函数"""
    csv_file_path = "/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/room_group_train_fix_clean_filtered_set_different_with_similarity.csv"

    # 创建处理器（默认包含分析）
    processor = RoomGroupingProcessor(include_analysis=True)

    print("开始处理200条数据（包含分析原因，使用多线程处理）...")
    try:
        output_file = processor.process_csv_file(
            csv_file_path,
            similarity_threshold=0.99,
            max_rows=50000,
            batch_size=20,
            max_workers=10
        )

        print(f"处理完成，输出文件: {output_file}")

        # 读取分块写入的最终结果进行统计显示
        print("\n=== 读取分块写入的最终结果 ===")
        final_df = pd.read_csv(output_file)
        print(f"最终结果文件包含 {len(final_df)} 条记录")

        # 显示结果示例
        print("\n=== 分析结果示例 ===")
        for i, row in final_df.head(5).iterrows():
            print(f"数据{i+1}:")
            print(f"  房型A: {row['spl_room_text']}")
            print(f"  房型B: {row['s_room_text']}")
            print(f"  判断: {row['llm_grouping_decision']}")
            print(f"  分析: {row['llm_grouping_analysis']}")
            print()

        # 显示最终统计信息
        yes_count = len(final_df[final_df['llm_grouping_decision'] == 'YES'])
        no_count = len(final_df[final_df['llm_grouping_decision'] == 'NO'])
        error_count = len(
            final_df[final_df['llm_grouping_decision'] == 'ERROR'])
        unknown_count = len(
            final_df[final_df['llm_grouping_decision'] == 'UNKNOWN'])
        total_count = len(final_df)

        print(f"\n=== 最终分组统计结果 ===")
        print(f"总共处理数据: {total_count} 条")
        print(f"YES (可以分组): {yes_count} 条 ({yes_count/total_count*100:.1f}%)")
        print(f"NO (不可以分组): {no_count} 条 ({no_count/total_count*100:.1f}%)")
        print(
            f"ERROR (处理错误): {error_count} 条 ({error_count/total_count*100:.1f}%)")
        print(
            f"UNKNOWN (未知): {unknown_count} 条 ({unknown_count/total_count*100:.1f}%)")
        if total_count > 0:
            print(f"成功处理率: {(yes_count + no_count)/total_count*100:.1f}%")

    except Exception as e:
        print(f"处理失败: {str(e)}")


if __name__ == "__main__":
    main()
