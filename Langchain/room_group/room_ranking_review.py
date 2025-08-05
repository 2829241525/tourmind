#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房型投诉分析处理器
读取processed_room_data.csv文件，对特定条件的数据进行客户投诉分析

功能：
- 读取processed_room_data.csv
- 对INCOMPARABLE且ratio>0.5的数据复制并置换room_a和room_b
- 去重处理
- 对HIGHER和INCOMPARABLE且ratio>0.5的数据调用LLM分析客户投诉情况
- 输出包含思考和结果的新CSV

使用方法：
python room_complaint_analysis.py
"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import pandas as pd
import numpy as np
import os
import sys
import logging
import time
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
from collections import defaultdict

# 添加文件锁，用于多线程安全写入
file_lock = threading.Lock()

# 优先使用本地langchain源码
LANGCHAIN_MASTER_PATH = "/home/maxon/disk2/roomMatch/room_match/Langchain/langchain-master"
sys.path.insert(0, os.path.join(LANGCHAIN_MASTER_PATH, "libs", "langchain"))
sys.path.insert(0, os.path.join(LANGCHAIN_MASTER_PATH, "libs", "core"))
sys.path.insert(0, os.path.join(
    LANGCHAIN_MASTER_PATH, "libs", "partners", "openai"))

# 基础配置
BASE_CONFIG = {
    'max_workers': 5,  # 最大并发线程数
    'batch_size': 20  # 每批处理的数据量
}

# LLM配置
LLM_CONFIG = {
    'qwen': {
        'model_name': 'qwen-plus-latest',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'temperature': 0.1,
        'max_tokens': 4000,
        'api_key': os.getenv('DASHSCOPE_API_KEY', 'sk-6d6b7b8eac764d08b90a817610f2f6ce')
    }
}

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('room_complaint_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RoomComplaintAnalyzer:
    """房型投诉分析器"""

    def __init__(self):
        """初始化分析器"""
        self.llm_config = LLM_CONFIG['qwen']
        self._setup_llm()
        self._setup_prompt()
        # 添加输出文件路径和已写入数据的追踪
        self.output_file = None
        self.written_data_cache = set()  # 缓存已写入的数据，避免重复

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
        prompt_template = """# 酒店住客投诉心理分析

你现在需要站在酒店客户的角度，模拟真实的客户心理和情绪反应。想象你是一位即将入住酒店的客户。

## 客户背景情境
- 你提前在酒店官网或预订平台上精心选择并支付了特定房型
- 你对预订的房型有一定的期待和要求
- 到达酒店办理入住时，被告知实际安排的房型与预订不符
- 你需要决定是否要为此向酒店投诉

## 客户心理分析框架

### 强烈投诉的情况 (COMPLAIN)：
1. **核心需求未满足**
   - 我预订时看中的关键特点在新房型中完全没有
   - 比如：我要海景房结果给了内庭房；我要套房结果给了标准间
   
2. **感觉被欺骗或降级**
   - 明显感觉新房型比我预订的要差
   - 觉得酒店在糊弄我或者故意降级
   - 怀疑酒店的诚信和服务质量

3. **情绪冲击强烈**
   - 期待落空，心情严重受影响
   - 觉得花了同样的钱却得到更差的服务
   - 担心整个住宿体验会很糟糕

### 可以接受不投诉的情况 (NO_COMPLAIN)：
1. **得到了免费升级感**
   - 新房型看起来不比原房型差，甚至更好
   - 感觉占到了便宜或得到了额外服务
   
2. **核心需求得到满足**
   - 虽然房型不同，但我最在意的功能都有
   - 整体感觉没有被亏待

3. **合理的平等替换**
   - 新房型与原房型价值相当
   - 能理解酒店的安排，没有明显的不公平感

## 需要分析的房型替换情况

{room_data}

## 分析要求

请你以真实客户的心态，对每组房型替换进行分析。对于每一组，请按以下格式输出：

```
组1: 原订房型[房型B] -> 实际房型[房型A]
思考：[站在客户角度分析：我预订房型B时最看重什么？实际给我房型A是否满足了我的核心期待？我会感觉被升级、平等对待还是被降级？我的情绪反应如何？]
结果：COMPLAIN 或 NO_COMPLAIN

组2: 原订房型[房型B] -> 实际房型[房型A]
思考：[继续以客户视角分析...]
结果：COMPLAIN 或 NO_COMPLAIN
```

## 注意事项
1. 必须以客户的真实心理进行换位思考
2. 重点分析客户的情绪感受和价值感知
3. 思考部分要体现客户的心理活动过程
4. 结果只能是 COMPLAIN 或 NO_COMPLAIN
5. 每组都必须有完整的思考和结果
"""
        self.prompt_template = prompt_template

    def load_processed_data(self, csv_file_path: str) -> pd.DataFrame:
        """加载处理过的房型数据"""
        try:
            logger.info(f"开始读取处理过的数据: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            logger.info(f"数据读取成功，总行数: {len(df)}")

            # 检查必要的列
            required_columns = ['room_a', 'room_b',
                                'comparison_result', 'ratio']
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")

            logger.info("数据列检查通过")
            return df

        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            raise

    def duplicate_and_swap_incomparable_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """复制INCOMPARABLE且ratio>0.5的数据并置换room_a和room_b"""
        try:
            # 筛选INCOMPARABLE且ratio>0.5的数据
            incomparable_high_ratio = df[
                (df['comparison_result'] == 'INCOMPARABLE') &
                (df['ratio'] > 0.5)
            ].copy()

            logger.info(
                f"找到 {len(incomparable_high_ratio)} 条INCOMPARABLE且ratio>0.5的数据")

            if len(incomparable_high_ratio) > 0:
                # 复制数据并置换room_a和room_b
                swapped_data = incomparable_high_ratio.copy()
                swapped_data['room_a'] = incomparable_high_ratio['room_b']
                swapped_data['room_b'] = incomparable_high_ratio['room_a']

                # 添加标记表示这是置换后的数据
                swapped_data['is_swapped'] = True
                df['is_swapped'] = False

                # 合并原数据和置换后的数据
                combined_df = pd.concat([df, swapped_data], ignore_index=True)
                logger.info(f"复制并置换数据完成，新增 {len(swapped_data)} 条数据")
                logger.info(f"合并后总数据量: {len(combined_df)}")

                return combined_df
            else:
                df['is_swapped'] = False
                logger.info("没有找到需要复制的数据")
                return df

        except Exception as e:
            logger.error(f"复制和置换数据失败: {str(e)}")
            raise

    def deduplicate_room_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """对room_a+room_b进行去重"""
        try:
            logger.info("开始进行room_a+room_b去重")
            original_count = len(df)

            # 创建组合键进行去重，考虑房型对的顺序无关性
            df['room_pair_key'] = df.apply(
                lambda row: tuple(
                    sorted([str(row['room_a']), str(row['room_b'])])),
                axis=1
            )

            # 去重，保留第一次出现的记录
            deduplicated_df = df.drop_duplicates(
                subset=['room_pair_key'], keep='first')

            # 删除临时列
            deduplicated_df = deduplicated_df.drop('room_pair_key', axis=1)

            final_count = len(deduplicated_df)
            removed_count = original_count - final_count

            logger.info(
                f"去重完成: {original_count} -> {final_count} (去除了 {removed_count} 条重复数据)")
            logger.info(f"剩余数据量: {final_count}")

            return deduplicated_df

        except Exception as e:
            logger.error(f"去重处理失败: {str(e)}")
            raise

    def filter_target_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """筛选需要分析的数据：HIGHER和INCOMPARABLE且ratio>0.5的数据"""
        try:
            # 筛选条件：comparison_result为HIGHER，或者为INCOMPARABLE且ratio>0.5
            target_data = df[
                (df['comparison_result'] == 'HIGHER') |
                ((df['comparison_result'] == 'INCOMPARABLE') & (df['ratio'] > 0.5))
            ].copy()

            logger.info(f"筛选目标数据完成:")
            logger.info(
                f"  HIGHER数据: {len(df[df['comparison_result'] == 'HIGHER'])}")
            logger.info(
                f"  INCOMPARABLE且ratio>0.5数据: {len(df[(df['comparison_result'] == 'INCOMPARABLE') & (df['ratio'] > 0.5)])}")
            logger.info(f"  总目标数据量: {len(target_data)}")

            return target_data

        except Exception as e:
            logger.error(f"筛选目标数据失败: {str(e)}")
            raise

    def prepare_batch_data(self, df: pd.DataFrame, batch_size: int = 20) -> List[List[Dict]]:
        """将数据分批准备"""
        try:
            batches = []
            total_rows = len(df)

            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                batch_data = []

                for idx, row in batch_df.iterrows():
                    batch_data.append({
                        'index': idx,
                        'room_a': row['room_a'],
                        'room_b': row['room_b'],
                        'comparison_result': row['comparison_result'],
                        'ratio': row['ratio'],
                        'is_swapped': row.get('is_swapped', False)
                    })

                batches.append(batch_data)

            logger.info(f"数据分批完成: {len(batches)} 个批次，每批最多 {batch_size} 条数据")
            return batches

        except Exception as e:
            logger.error(f"数据分批失败: {str(e)}")
            raise

    def call_llm_for_complaint_analysis(self, batch_data: List[Dict], batch_index: int) -> Tuple[int, str]:
        """调用LLM进行投诉分析"""
        try:
            # 构建房型数据文本 - 注意这里的顺序：room_b是原订，room_a是实际
            room_data_text = ""
            for i, data in enumerate(batch_data, 1):
                room_data_text += f"组{i}: 原订房型[{data['room_b']}] -> 实际房型[{data['room_a']}]\n"

            # 构建prompt
            prompt = self.prompt_template.format(room_data=room_data_text)

            messages = [HumanMessage(content=prompt)]
            result = self.llm.invoke(messages)

            llm_response = result.content if hasattr(
                result, 'content') else str(result)

            # 记录LLM返回的原始响应
            # logger.info(f"批次 {batch_index} LLM原始响应:")
            # logger.info(f"响应长度: {len(llm_response)} 字符")
            # logger.info(f"响应前500字符: {llm_response[:500]}...")

            return batch_index, llm_response

        except Exception as e:
            logger.error(f"LLM调用失败 (batch {batch_index}): {str(e)}")
            return batch_index, f"ERROR: {str(e)}"

    def parse_complaint_results(self, llm_response: str, batch_data: List[Dict]) -> List[Dict]:
        """解析LLM返回的投诉分析结果"""
        try:
            results = []
            lines = llm_response.split('\n')

            current_group = None
            current_thinking = ""
            current_result = ""

            logger.info(f"开始解析LLM响应，总行数: {len(lines)}")

            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                logger.debug(f"处理第{line_idx}行: {line[:100]}")

                # 匹配组号 - 更灵活的匹配模式
                if '组' in line and (':' in line or '：' in line):
                    # 如果有前一组的数据，先保存
                    if current_group is not None and current_thinking and current_result:
                        if current_group <= len(batch_data):
                            results.append({
                                'index': batch_data[current_group-1]['index'],
                                'thinking': current_thinking.strip(),
                                'result': current_result.strip()
                            })
                            logger.info(f"保存组{current_group}的分析结果")

                    # 开始新的一组
                    try:
                        # 更灵活的组号提取
                        import re
                        group_match = re.search(r'组(\d+)', line)
                        if group_match:
                            current_group = int(group_match.group(1))
                            current_thinking = ""
                            current_result = ""
                            logger.info(f"开始处理组{current_group}")
                        else:
                            logger.warning(f"无法从行中提取组号: {line}")
                            current_group = None
                    except Exception as e:
                        logger.warning(f"解析组号失败: {line}, 错误: {e}")
                        current_group = None

                # 匹配思考 - 支持中英文冒号和多行思考
                elif ('思考' in line) and ('：' in line or ':' in line):
                    if '：' in line:
                        thinking_content = line.split('：', 1)[1].strip()
                    else:
                        thinking_content = line.split(':', 1)[1].strip()
                    current_thinking = thinking_content
                    logger.debug(f"提取思考: {current_thinking[:50]}...")

                # 如果当前行不是新的标签但有当前思考，可能是思考的续行
                elif current_thinking and current_group is not None and not ('结果' in line) and not ('组' in line):
                    if not line.startswith(('思考', '结果')):
                        current_thinking += " " + line
                        logger.debug(f"追加思考内容: {line[:30]}...")

                # 匹配结果 - 支持中英文冒号
                elif ('结果' in line) and ('：' in line or ':' in line):
                    if '：' in line:
                        result_content = line.split('：', 1)[1].strip()
                    else:
                        result_content = line.split(':', 1)[1].strip()

                    # 确保结果是有效的 - 增强的结果提取逻辑
                    result_content = result_content.upper().strip()
                    if 'NO_COMPLAIN' in result_content or 'NO COMPLAIN' in result_content or '不会投诉' in result_content:
                        current_result = 'NO_COMPLAIN'
                    elif 'COMPLAIN' in result_content or '会投诉' in result_content or '投诉' in result_content:
                        current_result = 'COMPLAIN'
                    else:
                        logger.warning(f"无法识别的结果格式: {result_content}")
                        current_result = ""

                    logger.debug(f"提取结果: {current_result}")

            # 处理最后一组
            if current_group is not None and current_thinking and current_result:
                if current_group <= len(batch_data):
                    results.append({
                        'index': batch_data[current_group-1]['index'],
                        'thinking': current_thinking.strip(),
                        'result': current_result.strip()
                    })
                    logger.info(f"保存最后一组{current_group}的分析结果")

            logger.info(f"成功解析 {len(results)} 个投诉分析结果，期望 {len(batch_data)} 个")

            # 如果解析结果数量不匹配，记录详细信息并尝试修复
            if len(results) != len(batch_data):
                logger.warning(f"解析结果数量不匹配:")
                logger.warning(f"  期望: {len(batch_data)}")
                logger.warning(f"  实际: {len(results)}")
                logger.warning("LLM原始响应的前1000字符:")
                logger.warning(llm_response[:1000])

                # 尝试简单的行匹配修复
                if len(results) < len(batch_data):
                    logger.info("尝试简单的行匹配修复...")
                    self._try_simple_line_matching(
                        llm_response, batch_data, results)

            return results

        except Exception as e:
            logger.error(f"解析投诉分析结果失败: {str(e)}")
            return []

    def _try_simple_line_matching(self, llm_response: str, batch_data: List[Dict], results: List[Dict]) -> None:
        """尝试简单的行匹配来修复解析失败的结果"""
        try:
            lines = llm_response.split('\n')
            processed_indices = {r['index'] for r in results}

            # 找到包含COMPLAIN或NO_COMPLAIN的行
            for line in lines:
                line = line.strip().upper()
                if ('COMPLAIN' in line or '投诉' in line) and len(results) < len(batch_data):
                    # 找到一个还没有处理的数据项
                    for i, data in enumerate(batch_data):
                        if data['index'] not in processed_indices:
                            if 'NO_COMPLAIN' in line or '不会投诉' in line:
                                result_val = 'NO_COMPLAIN'
                            else:
                                result_val = 'COMPLAIN'

                            results.append({
                                'index': data['index'],
                                'thinking': "解析失败，使用简单匹配",
                                'result': result_val
                            })
                            processed_indices.add(data['index'])
                            logger.info(f"简单匹配修复: 组{i+1} -> {result_val}")
                            break

                    if len(results) >= len(batch_data):
                        break

        except Exception as e:
            logger.warning(f"简单行匹配修复失败: {e}")

    def process_and_write_batch(self, batch_data: List[Dict], batch_index: int, output_file: str, original_df: pd.DataFrame) -> Dict[str, int]:
        """处理一个批次并立即写入CSV文件"""
        try:
            logger.info(f"开始处理批次 {batch_index}，包含 {len(batch_data)} 条数据")

            # 调用LLM处理这个批次
            _, llm_response = self.call_llm_for_complaint_analysis(
                batch_data, batch_index)

            # 创建批次DataFrame副本
            batch_indices = [data['index'] for data in batch_data]
            batch_df = original_df.loc[batch_indices].copy()
            batch_df['thinking'] = ""
            batch_df['complaint_result'] = ""

            if not llm_response.startswith("ERROR:"):
                # 解析结果
                analysis_results = self.parse_complaint_results(
                    llm_response, batch_data)

                # 更新批次DataFrame
                for analysis_result in analysis_results:
                    idx = analysis_result['index']
                    if idx in batch_df.index:
                        batch_df.loc[idx,
                                     'thinking'] = analysis_result['thinking']
                        batch_df.loc[idx,
                                     'complaint_result'] = analysis_result['result']

                logger.info(
                    f"批次 {batch_index} 解析完成: {len(analysis_results)}/{len(batch_data)} 个分析结果")
            else:
                # 处理错误情况
                logger.error(f"批次 {batch_index} LLM调用失败: {llm_response}")
                batch_df['thinking'] = f"处理失败: {llm_response}"
                batch_df['complaint_result'] = "ERROR"

            # 使用文件锁安全写入CSV文件
            with file_lock:
                # 追加写入，不包含头部
                batch_df.to_csv(output_file, mode='a',
                                header=False, index=False, encoding='utf-8')
                logger.info(f"批次 {batch_index} 已安全写入CSV文件")

            # 统计本批次结果
            stats = {
                'complain': len(batch_df[batch_df['complaint_result'] == 'COMPLAIN']),
                'no_complain': len(batch_df[batch_df['complaint_result'] == 'NO_COMPLAIN']),
                'error': len(batch_df[batch_df['complaint_result'] == 'ERROR']),
                'processed': len(batch_df[batch_df['complaint_result'] != ""])
            }

            logger.info(
                f"批次 {batch_index} 处理完成 - 会投诉: {stats['complain']}, 不会投诉: {stats['no_complain']}, 错误: {stats['error']}")
            return stats

        except Exception as e:
            logger.error(f"处理批次 {batch_index} 失败: {str(e)}")

            # 返回错误统计
            error_stats = {
                'complain': 0,
                'no_complain': 0,
                'error': len(batch_data),
                'processed': len(batch_data)
            }

            # 尝试写入错误结果
            try:
                batch_indices = [data['index'] for data in batch_data]
                batch_df = original_df.loc[batch_indices].copy()
                batch_df['thinking'] = f'处理失败: {str(e)}'
                batch_df['complaint_result'] = 'ERROR'

                with file_lock:
                    batch_df.to_csv(output_file, mode='a',
                                    header=False, index=False, encoding='utf-8')
                    logger.info(f"批次 {batch_index} 错误结果已写入CSV文件")
            except Exception as write_error:
                logger.error(f"写入批次 {batch_index} 错误结果失败: {str(write_error)}")

            return error_stats

    def process_complaint_analysis(self, target_df: pd.DataFrame, output_file: str, max_workers: int = 5, batch_size: int = 20) -> Dict[str, int]:
        """处理投诉分析并分批次写入"""
        try:
            logger.info(f"开始投诉分析处理，数据量: {len(target_df)}")

            # 准备批次数据
            batches = self.prepare_batch_data(target_df, batch_size)

            # 创建输出文件并写入头部
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # 准备包含新列的DataFrame头部
            header_df = target_df.head(0).copy()
            header_df['thinking'] = ""
            header_df['complaint_result'] = ""
            header_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"创建输出文件并写入CSV头部: {output_file}")

            # 多线程批次处理
            total_count = len(target_df)
            batch_count = len(batches)
            logger.info(
                f"开始多线程批次处理模式，总数据: {total_count}条，批次大小: {batch_size}，批次数量: {batch_count}，最大线程数: {max_workers}")

            processed_batches = 0
            total_stats = {'complain': 0, 'no_complain': 0,
                           'error': 0, 'processed': 0}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有批次任务
                future_to_batch = {
                    executor.submit(self.process_and_write_batch, batch_data, batch_index, output_file, target_df): batch_index
                    for batch_index, batch_data in enumerate(batches)
                }

                # 处理完成的批次
                for future in as_completed(future_to_batch):
                    try:
                        batch_index = future_to_batch[future]
                        batch_stats = future.result()

                        # 累计统计信息
                        for key in total_stats:
                            total_stats[key] += batch_stats.get(key, 0)

                        processed_batches += 1
                        progress = (processed_batches / batch_count) * 100
                        logger.info(
                            f"批次 {batch_index} 处理并写入完成: {processed_batches}/{batch_count} ({progress:.1f}%)")

                    except Exception as e:
                        batch_index = future_to_batch[future]
                        logger.error(f"批次 {batch_index} 处理失败: {str(e)}")
                        processed_batches += 1

            logger.info(f"所有批次处理完成，结果已分批写入: {output_file}")

            # 统计最终结果
            logger.info("投诉分析结果统计:")
            logger.info(f"  总处理数据: {total_count}")
            logger.info(f"  成功处理: {total_stats['processed']}")
            logger.info(f"  会投诉: {total_stats['complain']}")
            logger.info(f"  不会投诉: {total_stats['no_complain']}")
            logger.info(f"  处理错误: {total_stats['error']}")

            return total_stats

        except Exception as e:
            logger.error(f"投诉分析处理失败: {str(e)}")
            raise

    def get_final_stats(self, output_file: str) -> Dict[str, int]:
        """从最终文件获取统计信息"""
        try:
            if os.path.exists(output_file):
                final_df = pd.read_csv(output_file)
                stats = {
                    'total_rows': len(final_df),
                    'processed': len(final_df[final_df['complaint_result'] != ""]),
                    'complain': len(final_df[final_df['complaint_result'] == 'COMPLAIN']),
                    'no_complain': len(final_df[final_df['complaint_result'] == 'NO_COMPLAIN']),
                    'error': len(final_df[final_df['complaint_result'] == 'ERROR'])
                }

                logger.info(f"最终文件统计:")
                logger.info(f"  总行数: {stats['total_rows']}")
                logger.info(f"  处理行数: {stats['processed']}")
                logger.info(f"  会投诉: {stats['complain']}")
                logger.info(f"  不会投诉: {stats['no_complain']}")
                logger.info(f"  处理错误: {stats['error']}")

                return stats
            else:
                logger.warning(f"输出文件不存在: {output_file}")
                return {}

        except Exception as e:
            logger.error(f"获取最终统计失败: {str(e)}")
            return {}

    def run_complaint_analysis(self, csv_file_path: str, max_workers: int = 5, batch_size: int = 20) -> str:
        """运行投诉分析

        Args:
            csv_file_path: CSV文件路径
            max_workers: 最大并发线程数
            batch_size: 每批处理的数据量

        Returns:
            输出文件路径
        """
        try:
            # 1. 加载数据
            logger.info("第1步：加载processed_room_data.csv")
            df = self.load_processed_data(csv_file_path)

            # 2. 复制并置换INCOMPARABLE且ratio>0.5的数据
            logger.info("第2步：复制并置换INCOMPARABLE且ratio>0.5的数据")
            df = self.duplicate_and_swap_incomparable_data(df)

            # 3. 去重处理
            logger.info("第3步：room_a+room_b去重处理")
            df = self.deduplicate_room_pairs(df)

            # 4. 筛选目标数据
            logger.info("第4步：筛选需要分析的数据")
            target_df = self.filter_target_data(df)

            if len(target_df) == 0:
                logger.warning("没有找到需要分析的数据")
                return None

            # 5. 准备输出文件
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/room_complaint_analysis_results_{timestamp}.csv"

            # 6. 进行投诉分析（每个批次完成后立即写入CSV）
            logger.info("第5步：进行投诉分析（分批次实时写入）")
            process_stats = self.process_complaint_analysis(
                target_df, output_file, max_workers, batch_size)

            # 7. 获取最终统计信息
            logger.info("第6步：获取最终统计信息")
            final_stats = self.get_final_stats(output_file)

            logger.info(f"✅ 投诉分析完成！输出文件: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"运行投诉分析失败: {str(e)}")
            raise


def main():
    """主函数"""
    csv_file_path = "/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/processed_room_data.csv"

    # 创建分析器
    analyzer = RoomComplaintAnalyzer()

    print("开始房型投诉分析...")
    print(f"输入文件: {csv_file_path}")

    # 获取配置
    max_workers = BASE_CONFIG['max_workers']
    batch_size = BASE_CONFIG['batch_size']

    print(f"使用 {max_workers} 个线程并发处理，每批次处理 {batch_size} 条数据")

    try:
        output_file = analyzer.run_complaint_analysis(
            csv_file_path=csv_file_path,
            max_workers=max_workers,
            batch_size=batch_size
        )

        if output_file:
            print(f"\n✅ 处理完成！")
            print(f"📁 输出文件: {output_file}")

            # 显示结果统计
            results_df = pd.read_csv(output_file)
            print(f"\n📊 处理统计:")
            print(f"总数据行数: {len(results_df)}")

            # 统计分析结果
            processed_df = results_df[results_df['complaint_result'] != ""]
            if len(processed_df) > 0:
                print(f"成功分析: {len(processed_df)}")
                print(f"投诉分析结果分布:")
                for result, count in processed_df['complaint_result'].value_counts().items():
                    percentage = (count / len(processed_df)) * 100
                    print(f"  {result}: {count} ({percentage:.1f}%)")

            # 显示一些示例结果
            print(f"\n🔍 示例分析结果:")
            sample_df = processed_df.head(3) if len(
                processed_df) > 0 else results_df.head(3)
            for i, row in sample_df.iterrows():
                print(f"\n示例 {i+1}:")
                print(f"原订房型: {row['room_b']}")
                print(f"实际房型: {row['room_a']}")
                print(f"比较结果: {row['comparison_result']}")
                if 'ratio' in row:
                    print(f"Ratio: {row['ratio']}")
                if row['complaint_result']:
                    print(f"投诉结果: {row['complaint_result']}")
                    print(f"分析思考: {row['thinking'][:100]}...")
                print("-" * 50)
        else:
            print("❌ 处理失败，请检查日志")

    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        logger.error(f"主程序执行失败: {str(e)}")


if __name__ == "__main__":
    main()
