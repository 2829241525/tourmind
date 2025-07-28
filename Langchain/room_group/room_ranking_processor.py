#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房型排序处理器
读取CSV文件，根据s_hotel_id分组获取房型信息
调用LLM对房型进行等级排序

功能：
- 根据country_code比例选取1000个酒店
- 按酒店分组获取房型描述
- 调用大模型进行房型两两比较和排序
- 输出房型等级关系（高于/低于/不相关/相等）

使用方法：
python room_ranking_processor.py
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
    # 指定要处理的酒店ID列表，如果为None则按比例采样 [12395035, 25709125, 21109218]
    'hotel_ids': None,
    # 'hotel_ids': [12976637],
    'target_hotel_count': 3000,  # 采样目标酒店数量（当hotel_ids为None时使用）
    'max_workers': 5,  # 最大并发线程数
    'batch_size': 20  # 每批处理的房型对数量
}

# LLM配置qwen-plus-latest deepseek-r1
LLM_CONFIG = {
    'qwen': {
        'model_name': 'qwen-plus-latest',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'temperature': 0.1,
        'max_tokens': 2000,
        'api_key': os.getenv('DASHSCOPE_API_KEY', 'sk-6d6b7b8eac764d08b90a817610f2f6ce')
    }
}

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('room_ranking.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RoomRankingProcessor:
    """房型排序处理器"""

    def __init__(self):
        """初始化处理器"""
        self.llm_config = LLM_CONFIG['qwen']
        self._setup_llm()
        self._setup_prompt()
        # 添加输出文件路径和已写入数据的追踪
        self.output_file = None
        self.written_data_cache = set()  # 缓存已写入的数据，避免重复

    def _load_existing_data(self, output_file: str) -> set:
        """加载已存在的CSV数据，用于去重检查"""
        if not os.path.exists(output_file):
            return set()

        try:
            existing_df = pd.read_csv(output_file)
            written_data = set()

            for _, row in existing_df.iterrows():
                # 创建唯一标识符：酒店ID + 房型A + 房型B + 比较结果
                key = (
                    row['s_hotel_id'],
                    row['room_a'],
                    row['room_b'],
                    row['comparison_result']
                )
                written_data.add(key)

            logger.info(f"加载了 {len(written_data)} 条已存在的比较记录")
            return written_data

        except Exception as e:
            logger.error(f"加载已存在数据失败: {str(e)}")
            return set()

    def _is_data_written(self, hotel_id: int, room_a: str, room_b: str, comparison_result: str) -> bool:
        """检查数据是否已经写入过"""
        key = (hotel_id, room_a, room_b, comparison_result)
        reverse_key = (hotel_id, room_b, room_a, comparison_result)  # 考虑房型对的顺序

        return key in self.written_data_cache or reverse_key in self.written_data_cache

    def save_single_hotel_results(self, hotel_id: int, country_code: str,
                                  comparison_results: List[Dict], output_file: str):
        """保存单个酒店的排序结果（增量写入）"""
        try:
            with file_lock:  # 确保线程安全
                # 准备新数据
                new_data = []

                if comparison_results:
                    for comparison in comparison_results:
                        room_a = comparison['room_a']
                        room_b = comparison['room_b']
                        result = comparison['comparison_result']

                        # 检查是否已经写入过
                        if not self._is_data_written(hotel_id, room_a, room_b, result):
                            new_data.append({
                                's_hotel_id': hotel_id,
                                'country_code': country_code,
                                'room_a': room_a,
                                'room_b': room_b,
                                'comparison_result': result,
                                'reason': comparison['reason']
                            })

                            # 更新缓存
                            key = (hotel_id, room_a, room_b, result)
                            self.written_data_cache.add(key)
                else:
                    # 如果没有比较结果，记录错误（但也要检查是否已写入）
                    if not self._is_data_written(hotel_id, '', '', 'ERROR'):
                        new_data.append({
                            's_hotel_id': hotel_id,
                            'country_code': country_code,
                            'room_a': '',
                            'room_b': '',
                            'comparison_result': 'ERROR',
                            'reason': '处理失败或无比较结果'
                        })

                        # 更新缓存
                        key = (hotel_id, '', '', 'ERROR')
                        self.written_data_cache.add(key)

                # 如果有新数据，写入文件
                if new_data:
                    new_df = pd.DataFrame(new_data)

                    # 检查文件是否存在，决定是追加还是创建
                    if os.path.exists(output_file):
                        # 追加写入
                        new_df.to_csv(output_file, mode='a',
                                      header=False, index=False, encoding='utf-8')
                        logger.info(
                            f"✅ 酒店 {hotel_id}: 追加写入 {len(new_data)} 条新比较记录")
                    else:
                        # 创建新文件
                        new_df.to_csv(output_file, mode='w',
                                      header=True, index=False, encoding='utf-8')
                        logger.info(
                            f"✅ 酒店 {hotel_id}: 创建文件并写入 {len(new_data)} 条比较记录")
                else:
                    logger.info(f"📋 酒店 {hotel_id}: 所有数据已存在，跳过写入")

        except Exception as e:
            logger.error(f"保存酒店 {hotel_id} 结果失败: {str(e)}")
            raise

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
        prompt_template = """# 第一部分：角色与任务定义 (Role & Task Definition)

你现在是一名极其严谨、注重细节的酒店数据分析专家。你的核心任务是根据房型描述，对它们进行两两比较，并严格依据下述的“全面优势判断”原则，分析其相对等级。

**任务**: 对房型列表中的每一个房型对进行分析，判断第一个房型相对于第二个房型的等级关系（更高、更低、相等、或无法比较）。

# 第二部分：核心分析原则 (Core Analytical Principles)

## 一、比较维度与内部层级 (Dimensions of Comparison & Hierarchies)

在分析时，你需要从以下所有维度对每个房型进行特征提取和评估，如果出现以下维度所列以外的词汇，请以自身所具备的知识进行分析，以下知识仅做参考。

1.  **房型等级 (Room Grade)**: Executive > Premier > Deluxe > Superior > Standard > Economy. (默认: Standard)
2.  **房间类型 (Room Type)**: Villa / Apartment /family > Suite > Studio > Room.
3.  **景观 (View)**:
    *   **有无对比**: 任何明确指出的景观 > 无景观。
    *   **同类景观内部层级**: Beachfront / Oceanfront > Sea View; Pool Access > Pool View.
    *   **不同景观类型不可比**: 不同类型的景观（如 海景 vs. 城景）视为特色不同，没有高低之分。

4.  **核心设施 (Key Facilities)**: 拥有 Balcony, Kitchen, Jacuzzi 等设施的房型更优。**无障碍设施(Accessible)有特殊规则，见下文**。
5.  **床数量 (Bed Count)**: 床的数量越多越优 (e.g., 2 beds > 1 bed).
6.  **床尺寸 (Bed Size)**:
    *   **基础层级**: King > Queen > Double > Single/Twin.
    *   **特殊规则：不确定的床型（"或"逻辑）**: 描述中包含“/”或“或”表示床型不确定，代表一个可能性范围。其等级基于与确定床型的对比：
        *   **低于 "确定的最优选项"**: `1 Queen Bed/1 King Bed` **低于** `1 King Bed`。因为前者不保证能获得King Bed。
        *   **高于 "确定的最差选项"**: `1 Queen Bed/1 King Bed` **高于** `1 Queen Bed`。因为前者有获得更优King Bed的可能性。

7.  **房间主题/特色 (room theme)**: 
    *   **描述分析**: 分析整体描述，哪些描述词汇是房间的主题/特色，主题/特色不一定以theme/feature表述。
    *   **有无对比**: 有主题/特色描述 > 无主题/特色描述。
    *   ** 不同主题/特色不可比**: 不同类型的主题/特色（如 动漫/电影/明星/特殊景观/动物等等）视为主题/特色不同，没有高低之分。当两个房型主题不同时，强制不可比
7.  **附加福利 (Benefits)**: 包含免费早餐、机场接送、行政酒廊待遇等福利的房型更优。
8.  **限制条款 (Restrictions)**: “可退款”优于“不可退款”，可吸烟优于禁烟，可携带宠物优于禁带宠物。

## 二、判断规则 (Rules of Judgment)

请严格遵循以下规则来判定房型对的关系。

### HIGHER (A 全面优于 B)
*   **规则**: 房型A比房型B **HIGHER**，当且仅当：
    1.  房型A在**至少一个**比较维度上明确优于房型B。
    2.  并且，房型A在**所有其他**比较维度上**等于或优于**房型B（即，B没有任何一个维度优于A）。

### LOWER (B 全面优于 A)
*   **规则**: 房型A比房型B **LOWER**，当且仅当：
    1.  房型B在**至少一个**比较维度上明确优于房型A。
    2.  并且，房型B在**所有其他**比较维度上**等于或优于**房型A（即，A没有任何一个维度优于B）。

### INCOMPARABLE (优劣互换、特色不同 或 强制不可比)
*   **规则**: 如果不满足 HIGHER, LOWER, 或 EQUAL 的任何条件，则为 **INCOMPARABLE**。这通常是因为：
    *   **强制不可比**: 一个房型是 `Accessible` 而另一个不是。
    *   **特色不同**: 拥有不同类型的景观 (`海景` vs `城景`)、不同的核心设施 (`阳台` vs `厨房`)、或不同的福利 (`早餐` vs `机场接送`)。
    *   **优劣互换 (关键维度冲突)**: A在某些维度优于B，同时B在另一些维度优于A (`A: Deluxe Room` vs `B: Standard Suite`; `A: 1 King Bed` vs `B: 2 Double Beds`)。
    *   **特别关注**: A在某些维度优于B，但B在以上任何一个维度优于A（包括床数，设施，景观等等）。则判定为 **INCOMPARABLE**。
    *   **主题/特征**: A与B都有各自的主题,且主题不一致时。则判定为 **INCOMPARABLE**。

### EQUAL (完全相同)
*   **规则**: 只有当两个房型在**所有**比较维度上都完全相同时，才判定为 **EQUAL**。

# 第三部分：分析思维框架 (Analytical Framework)

为了确保结果的准确性，请严格遵循以下思考步骤：

1.  **选取一对房型 (A, B)** 从列表中。

2.  **全面特征提取**: 首先，并列分析A和B的所有特征，识别出它们在所有维度上的具体情况。

3.  **进行优劣势对比 (核心逻辑)**:
    *   **第一步：检查强制不可比条件**。检查“无障碍设施”是否不同。同时检查是否存在“特色不同”（如不同景观、不同福利）。如果存在，立即判定为 **INCOMPARABLE** 并说明原因。
    *   **第二步：评估双方优势**。如果上一步通过，则系统地比较所有剩余维度：
        *   记录下所有 **“A优于B”** 的维度。
        *   记录下所有 **“B优于A”** 的维度。
    *   **第三步：应用最终判断规则**:
        *   如果 **“A优于B”的维度列表不为空** 且 **“B优于A”的维度列表为空** -> 结果是 **HIGHER**。
        *   如果 **“A优于B”的维度列表为空** 且 **“B优于A”的维度列表不为空** -> 结果是 **LOWER**。
        *   如果 **“A优于B”的维度列表为空** 且 **“B优于A”的维度列表也为空** -> 结果是 **EQUAL**。
        *   如果 **“A优于B”的维度列表不为空** 且 **“B优于A”的维度列表也不为空** -> 存在优劣互换，结果是 **INCOMPARABLE**。

4.  **反思逻辑**，如果输出不是INCOMPARABLE需要进行二次思考，对输出的结果进行再次思考，输出是否综合考虑而忽略了规则，输出格式是否合规等，注明：一定要有房间主题/特色规则的相关分析。
5.  **循环操作**，直到完成所有房型对的比较。

# 第四部分：分析任务与输出格式

**房型对比列表:**
{room_pairs}

**输出格式要求:**
请严格按照以下格式输出，对上述每一个房型对比进行分析。每个比较都必须另起一行，并给出具体、清晰的理由。

## 比较结果
房型A名称 vs 房型B名称: - 理由：[A在某个维度上更优（说明是哪个维度），且B房型、床型、景观、设施、福利、特色等其他领域上没有任何优势] - 反思逻辑：[和之前思考一致/仔细思考后发现...] - 最终结论：**HIGHER**
房型A名称 vs 房型B名称: - 理由：[A在某个维度上更劣（说明是哪个维度），且B房型、床型、景观、设施、福利、特色等其他领域上没有任何优势] - 反思逻辑：[和之前思考一致/仔细思考后发现...] - 最终结论：**LOWER**
房型A名称 vs 房型B名称: - 理由：[所有包括型、床型、景观、设施、福利、特色等维度均完全相同] - 反思逻辑：[和之前思考一致/仔细思考后发现...] - 最终结论：**EQUAL**
房型A名称 vs 房型B名称: - 理由：[存在优劣互换或特色不同，请明确指出原因，例如：无障碍设施不一致；或关键维度冲突(A在等级上占优，B在床数量上占优)] - 反思逻辑：[和之前思考一致/仔细思考后发现...] - 最终结论：**INCOMPARABLE**
"""
        self.prompt_template = prompt_template

    def calculate_expected_comparisons(self, room_count: int) -> int:
        """
        计算期望的房型比较数量

        Args:
            room_count: 房型数量

        Returns:
            期望的比较数量 (组合数 C(n,2) = n*(n-1)/2)
        """
        if room_count < 2:
            return 0
        return room_count * (room_count - 1) // 2

    def validate_comparison_count(self, hotel_id: int, room_list: List[str],
                                  comparison_results: List[Dict]) -> Dict[str, any]:
        """
        校验房型比较结果的数量完整性

        Args:
            hotel_id: 酒店ID
            room_list: 房型列表
            comparison_results: 比较结果列表

        Returns:
            校验结果字典，包含统计信息
        """
        room_count = len(room_list)
        expected_count = self.calculate_expected_comparisons(room_count)
        actual_count = len(comparison_results)

        # 统计各种比较结果类型
        result_stats = {}
        missing_pairs = []

        if comparison_results:
            # 统计结果类型分布
            for result in comparison_results:
                result_type = result.get('comparison_result', 'UNKNOWN')
                result_stats[result_type] = result_stats.get(
                    result_type, 0) + 1

            # 检查缺失的房型对
            found_pairs = set()
            for result in comparison_results:
                room_a = result.get('room_a', '')
                room_b = result.get('room_b', '')
                # 标准化房型对（确保一致性）
                pair = tuple(sorted([room_a, room_b]))
                found_pairs.add(pair)

            # 生成所有可能的房型对
            expected_pairs = set()
            for i in range(len(room_list)):
                for j in range(i + 1, len(room_list)):
                    pair = tuple(sorted([room_list[i], room_list[j]]))
                    expected_pairs.add(pair)

            # 找出缺失的房型对
            missing_pairs = list(expected_pairs - found_pairs)

        validation_result = {
            'hotel_id': hotel_id,
            'room_count': room_count,
            'expected_comparisons': expected_count,
            'actual_comparisons': actual_count,
            'is_complete': actual_count == expected_count,
            'completion_rate': (actual_count / expected_count * 100) if expected_count > 0 else 0,
            'result_stats': result_stats,
            'missing_pairs_count': len(missing_pairs),
            'missing_pairs': missing_pairs[:5]  # 只记录前5个缺失的对
        }

        # 输出详细日志
        logger.info(f"🏨 酒店 {hotel_id} 比较结果校验:")
        logger.info(f"  📊 房型数量: {room_count}")
        logger.info(f"  🎯 期望比较数: {expected_count}")
        logger.info(f"  ✅ 实际比较数: {actual_count}")
        logger.info(f"  📈 完成率: {validation_result['completion_rate']:.1f}%")

        if validation_result['is_complete']:
            logger.info(f"  ✅ 比较完整性: 完整")
        else:
            logger.warning(
                f"  ⚠️  比较完整性: 不完整，缺少 {expected_count - actual_count} 条比较")
            if missing_pairs:
                logger.warning(f"  🔍 部分缺失的房型对: {missing_pairs[:3]}")

        if result_stats:
            logger.info(f"  📋 结果类型分布:")
            for result_type, count in result_stats.items():
                percentage = (count / actual_count *
                              100) if actual_count > 0 else 0
                logger.info(f"    {result_type}: {count} ({percentage:.1f}%)")

        return validation_result

    def log_overall_validation_summary(self, all_validations: List[Dict]):
        """
        输出整体校验结果摘要

        Args:
            all_validations: 所有酒店的校验结果列表
        """
        if not all_validations:
            logger.warning("📊 没有校验结果可以汇总")
            return

        total_hotels = len(all_validations)
        complete_hotels = sum(1 for v in all_validations if v['is_complete'])
        total_expected = sum(v['expected_comparisons']
                             for v in all_validations)
        total_actual = sum(v['actual_comparisons'] for v in all_validations)

        # 统计完成率分布
        completion_rates = [v['completion_rate'] for v in all_validations]
        avg_completion_rate = sum(completion_rates) / \
            len(completion_rates) if completion_rates else 0

        # 统计结果类型分布
        overall_result_stats = {}
        for validation in all_validations:
            for result_type, count in validation['result_stats'].items():
                overall_result_stats[result_type] = overall_result_stats.get(
                    result_type, 0) + count

        logger.info("=" * 60)
        logger.info("📊 整体校验结果摘要")
        logger.info("=" * 60)
        logger.info(f"🏨 总处理酒店数: {total_hotels}")
        logger.info(
            f"✅ 完整比较酒店数: {complete_hotels} ({complete_hotels/total_hotels*100:.1f}%)")
        logger.info(
            f"⚠️  不完整比较酒店数: {total_hotels - complete_hotels} ({(total_hotels-complete_hotels)/total_hotels*100:.1f}%)")
        logger.info(f"🎯 总期望比较数: {total_expected}")
        logger.info(f"✅ 总实际比较数: {total_actual}")
        logger.info(
            f"📈 整体完成率: {total_actual/total_expected*100:.1f}%" if total_expected > 0 else "0%")
        logger.info(f"📊 平均完成率: {avg_completion_rate:.1f}%")

        if overall_result_stats:
            logger.info("📋 整体结果类型分布:")
            for result_type, count in sorted(overall_result_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_actual *
                              100) if total_actual > 0 else 0
                logger.info(f"  {result_type}: {count} ({percentage:.1f}%)")

        # 按完成率分组统计
        perfect_count = sum(
            1 for v in all_validations if v['completion_rate'] == 100)
        good_count = sum(1 for v in all_validations if 80 <=
                         v['completion_rate'] < 100)
        fair_count = sum(1 for v in all_validations if 50 <=
                         v['completion_rate'] < 80)
        poor_count = sum(
            1 for v in all_validations if v['completion_rate'] < 50)

        logger.info("📈 完成率分布:")
        logger.info(f"  🌟 完美 (100%): {perfect_count} 酒店")
        logger.info(f"  ✅ 良好 (80-99%): {good_count} 酒店")
        logger.info(f"  ⚠️  一般 (50-79%): {fair_count} 酒店")
        logger.info(f"  ❌ 较差 (<50%): {poor_count} 酒店")
        logger.info("=" * 60)

    def load_and_sample_hotels(self, csv_file_path: str, target_hotel_count: int = 1000, hotel_ids: List[int] = None) -> pd.DataFrame:
        """加载数据并按国家比例采样酒店

        Args:
            csv_file_path: CSV文件路径
            target_hotel_count: 目标采样酒店数量（当hotel_ids为None时使用）
            hotel_ids: 指定要处理的酒店ID列表，如果提供则直接处理这些酒店

        Returns:
            包含采样或指定酒店数据的DataFrame
        """
        try:
            logger.info(f"开始读取CSV文件: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            logger.info(f"CSV文件读取成功，总行数: {len(df)}")

            if hotel_ids:
                # 如果指定了酒店ID，直接筛选这些酒店的数据
                sampled_df = df[df['s_hotel_id'].isin(hotel_ids)].copy()
                found_hotels = sampled_df['s_hotel_id'].unique()
                missing_hotels = set(hotel_ids) - set(found_hotels)

                logger.info(f"指定处理 {len(hotel_ids)} 个酒店")
                logger.info(f"找到 {len(found_hotels)} 个酒店的数据")
                if missing_hotels:
                    logger.warning(f"未找到以下酒店的数据: {missing_hotels}")

                # 显示找到的酒店的国家分布
                country_stats = sampled_df.groupby(
                    's_hotel_id')['country_code'].first().value_counts()
                logger.info("找到酒店的国家分布:")
                for country, count in country_stats.items():
                    logger.info(f"  {country}: {count}")

                return sampled_df

            # 如果没有指定酒店ID，按原有逻辑进行采样
            # 获取每个国家的酒店数量
            hotel_country = df.groupby('s_hotel_id')[
                'country_code'].first().reset_index()
            country_hotel_counts = hotel_country['country_code'].value_counts()
            total_hotels = len(hotel_country)

            logger.info(f"总酒店数: {total_hotels}")
            logger.info("各国酒店分布:")
            for country, count in country_hotel_counts.head(10).items():
                percentage = (count / total_hotels) * 100
                logger.info(f"  {country}: {count} ({percentage:.1f}%)")

            # 按比例采样酒店
            sampled_hotels = []
            for country, count in country_hotel_counts.items():
                proportion = count / total_hotels
                sample_size = max(1, int(target_hotel_count * proportion))

                country_hotels = hotel_country[hotel_country['country_code']
                                               == country]['s_hotel_id'].tolist()
                if len(country_hotels) >= sample_size:
                    sampled = np.random.choice(
                        country_hotels, sample_size, replace=False)
                else:
                    sampled = country_hotels

                sampled_hotels.extend(sampled)

            # 如果采样数量不够目标数量，补充采样
            if len(sampled_hotels) < target_hotel_count:
                remaining_hotels = hotel_country[~hotel_country['s_hotel_id'].isin(
                    sampled_hotels)]['s_hotel_id'].tolist()
                additional_needed = target_hotel_count - len(sampled_hotels)
                if len(remaining_hotels) >= additional_needed:
                    additional = np.random.choice(
                        remaining_hotels, additional_needed, replace=False)
                    sampled_hotels.extend(additional)
                else:
                    sampled_hotels.extend(remaining_hotels)

            # 限制到目标数量
            sampled_hotels = sampled_hotels[:target_hotel_count]

            logger.info(f"最终采样酒店数: {len(sampled_hotels)}")

            # 筛选出采样酒店的数据
            sampled_df = df[df['s_hotel_id'].isin(sampled_hotels)].copy()
            logger.info(f"采样后数据行数: {len(sampled_df)}")

            return sampled_df

        except Exception as e:
            logger.error(f"加载和采样数据失败: {str(e)}")
            raise

    def deduplicate_room_descriptions(self, room_descriptions: List[str]) -> List[str]:
        """
        房型描述去重处理

        Args:
            room_descriptions: 原始房型描述列表

        Returns:
            去重后的房型描述列表
        """
        if not room_descriptions:
            return []

            # 简单标准化处理函数
        def normalize_room_desc(desc: str) -> str:
            """简单标准化房型描述，用于去重比较"""
            if not desc:
                return ""
            # 转小写，去除首尾空格，统一空格
            normalized = desc.lower().strip()
            normalized = ' '.join(normalized.split())
            return normalized

        # 使用字典记录标准化描述到原始描述的映射
        normalized_to_original = {}
        seen_normalized = set()

        for desc in room_descriptions:
            normalized = normalize_room_desc(desc)

            if normalized and normalized not in seen_normalized:
                seen_normalized.add(normalized)
                normalized_to_original[normalized] = desc

        deduplicated = list(normalized_to_original.values())

        # 记录去重情况
        original_count = len(room_descriptions)
        final_count = len(deduplicated)

        if original_count != final_count:
            logger.info(
                f"  房型去重: {original_count} -> {final_count} (去除了 {original_count - final_count} 个重复项)")

            # 显示被去除的重复项（如果有的话）
            if logger.isEnabledFor(logging.DEBUG):
                original_normalized = [normalize_room_desc(
                    desc) for desc in room_descriptions]
                duplicates = [desc for desc in room_descriptions
                              if original_normalized.count(normalize_room_desc(desc)) > 1]
                if duplicates:
                    logger.debug(f"  重复房型: {list(set(duplicates))}")

        return deduplicated

    def prepare_hotel_room_data(self, df: pd.DataFrame) -> Dict[int, Dict]:
        """准备酒店房型数据"""
        try:
            hotel_room_data = {}
            total_original_rooms = 0
            total_final_rooms = 0

            for hotel_id in df['s_hotel_id'].unique():
                hotel_df = df[df['s_hotel_id'] == hotel_id]
                country = hotel_df['country_code'].iloc[0]

                # 组合房型描述
                room_descriptions = []
                for _, row in hotel_df.iterrows():
                    room_name = str(row['room_name']) if pd.notna(
                        row['room_name']) else ""
                    bed_type = str(row['bed_type_desc']) if pd.notna(
                        row['bed_type_desc']) else ""

                    # 用空格组合，去除多余空格
                    combined = f"{room_name} {bed_type}".strip()
                    if combined:  # 只要非空就添加，后续统一去重
                        room_descriptions.append(combined)

                original_count = len(room_descriptions)
                total_original_rooms += original_count

                # 去重处理
                room_descriptions = self.deduplicate_room_descriptions(
                    room_descriptions)
                final_count = len(room_descriptions)
                total_final_rooms += final_count

                # 只处理有多个房型的酒店（去重后）
                if len(room_descriptions) >= 2:
                    hotel_room_data[hotel_id] = {
                        'country': country,
                        'rooms': room_descriptions
                    }

            # 整体统计
            logger.info(f"📊 数据准备完成:")
            logger.info(f"  总去重后房型数: {total_final_rooms}")
            logger.info(
                f"  去重比例: {(total_original_rooms - total_final_rooms)/total_original_rooms*100:.1f}%" if total_original_rooms > 0 else "0%")
            logger.info(f"  准备了 {len(hotel_room_data)} 个酒店的房型数据")

            return hotel_room_data

        except Exception as e:
            logger.error(f"准备房型数据失败: {str(e)}")
            raise

    def generate_room_pairs(self, room_list: List[str]) -> List[Tuple[str, str]]:
        """
        生成所有房型对比组合

        Args:
            room_list: 房型列表

        Returns:
            房型对比元组列表
        """
        room_pairs = []
        for i in range(len(room_list)):
            for j in range(i + 1, len(room_list)):
                room_pairs.append((room_list[i], room_list[j]))
        return room_pairs

    def split_pairs_into_batches(self, room_pairs: List[Tuple[str, str]], batch_size: int = 20) -> List[List[Tuple[str, str]]]:
        """
        将房型对分批处理

        Args:
            room_pairs: 房型对列表
            batch_size: 每批最大数量

        Returns:
            分批后的房型对列表
        """
        batches = []
        for i in range(0, len(room_pairs), batch_size):
            batch = room_pairs[i:i + batch_size]
            batches.append(batch)
        return batches

    def call_llm_for_room_pairs(self, hotel_id: int, room_pairs: List[Tuple[str, str]], batch_index: int = 0) -> Tuple[int, str, int]:
        """调用LLM对房型对进行比较"""
        try:
            # 构建房型对比文本
            room_pairs_text = ""
            for i, (room_a, room_b) in enumerate(room_pairs, 1):
                room_pairs_text += f"{i}. {room_a} vs {room_b}\n"

            # 构建prompt
            prompt = self.prompt_template.format(
                room_pairs=room_pairs_text
            )

            messages = [HumanMessage(content=prompt)]
            result = self.llm.invoke(messages)

            llm_response = result.content if hasattr(
                result, 'content') else str(result)
            return hotel_id, llm_response, batch_index

        except Exception as e:
            logger.error(
                f"LLM调用失败 (hotel_id {hotel_id}, batch {batch_index}): {str(e)}")
            return hotel_id, f"ERROR: {str(e)}", batch_index

    def parse_comparison_results(self, llm_response: str, room_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """解析LLM输出的房型比较结果"""
        try:
            results = []
            lines = llm_response.split('\n')

            # 创建房型对的映射，用于验证
            expected_pairs = set()
            for room_a, room_b in room_pairs:
                # 标准化房型对（确保一致性）
                pair = tuple(sorted([room_a.strip(), room_b.strip()]))
                expected_pairs.add(pair)

            for line in lines:
                line = line.strip()
                if ' vs ' in line and '最终结论：**' in line:
                    try:
                        # 解析新格式：房型A名称 vs 房型B名称: - 理由：[...] - 反思逻辑：[...] - 最终结论：**RESULT**
                        # 分离房型对和内容部分
                        if ': - 理由：' in line:
                            room_pair_part, content_part = line.split(
                                ': - 理由：', 1)
                        else:
                            continue

                        # 提取房型名称
                        if ' vs ' in room_pair_part:
                            room_a, room_b = room_pair_part.split(' vs ', 1)
                            room_a = room_a.strip()
                            room_b = room_b.strip()
                        else:
                            continue

                        # 解析内容部分：理由 - 反思逻辑 - 最终结论：**RESULT**
                        reason = ""
                        reflection = ""
                        result = ""

                        if ' - 反思逻辑：' in content_part and ' - 最终结论：**' in content_part:
                            # 完整格式：理由 - 反思逻辑 - 最终结论
                            reason_part, remaining = content_part.split(
                                ' - 反思逻辑：', 1)
                            reason = reason_part.strip()

                            if ' - 最终结论：**' in remaining:
                                reflection_part, conclusion_part = remaining.split(
                                    ' - 最终结论：**', 1)
                                reflection = reflection_part.strip()

                                # 提取结果
                                if '**' in conclusion_part:
                                    result = conclusion_part.split(
                                        '**')[0].strip()
                                else:
                                    result = conclusion_part.strip()

                        elif ' - 最终结论：**' in content_part:
                            # 简化格式：理由 - 最终结论（可能没有反思逻辑）
                            reason_part, conclusion_part = content_part.split(
                                ' - 最终结论：**', 1)
                            reason = reason_part.strip()

                            # 提取结果
                            if '**' in conclusion_part:
                                result = conclusion_part.split('**')[0].strip()
                            else:
                                result = conclusion_part.strip()

                        # 清理处理理由和反思逻辑
                        if reason.startswith('[') and reason.endswith(']'):
                            reason = reason[1:-1].strip()
                        if reflection.startswith('[') and reflection.endswith(']'):
                            reflection = reflection[1:-1].strip()

                        # 合并理由和反思逻辑
                        combined_reason = reason
                        if reflection:
                            combined_reason += f" | 反思：{reflection}"

                        # 验证结果是否为有效的比较结果
                        if result.upper() in ['HIGHER', 'LOWER', 'EQUAL', 'INCOMPARABLE']:
                            result = result.upper()

                            # 验证房型对是否在期望列表中
                            parsed_pair = tuple(sorted([room_a, room_b]))
                            if parsed_pair in expected_pairs:
                                results.append({
                                    'room_a': room_a,
                                    'room_b': room_b,
                                    'comparison_result': result,
                                    'reason': combined_reason
                                })
                            else:
                                # 尝试模糊匹配
                                found_match = False
                                for expected_a, expected_b in room_pairs:
                                    if ((room_a in expected_a or expected_a in room_a) and
                                        (room_b in expected_b or expected_b in room_b)) or \
                                       ((room_a in expected_b or expected_b in room_a) and
                                            (room_b in expected_a or expected_a in room_b)):
                                        results.append({
                                            'room_a': room_a,
                                            'room_b': room_b,
                                            'comparison_result': result,
                                            'reason': combined_reason
                                        })
                                        found_match = True
                                        break

                                if not found_match:
                                    logger.warning(
                                        f"房型对未找到匹配: {room_a} vs {room_b}")
                        else:
                            logger.warning(f"无效的比较结果: {result}")

                    except Exception as e:
                        logger.warning(f"解析比较结果行失败: {line}, 错误: {str(e)}")
                        continue

            logger.info(f"成功解析 {len(results)} 个房型比较结果，期望 {len(room_pairs)} 个")
            return results

        except Exception as e:
            logger.error(f"解析比较结果失败: {str(e)}")
            return []

    def process_hotel_rankings(self, hotel_room_data: Dict[int, Dict], max_workers: int = 5, batch_size: int = 20) -> Tuple[Dict[int, List[Dict]], List[Dict]]:
        """处理酒店房型排序"""
        try:
            logger.info(
                f"开始处理 {len(hotel_room_data)} 个酒店的房型排序，批次大小: {batch_size}")

            # 初始化输出文件
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_file = f"/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/room_ranking_results_{timestamp}.csv"

            # 加载已存在的数据（如果有）
            self.written_data_cache = self._load_existing_data(
                self.output_file)

            results = {}
            all_validations = []
            processed_hotels = 0

            # 为每个酒店生成批次任务
            all_tasks = []
            for hotel_id, data in hotel_room_data.items():
                room_list = data['rooms']
                room_pairs = self.generate_room_pairs(room_list)
                batches = self.split_pairs_into_batches(room_pairs, batch_size)

                logger.info(
                    f"酒店 {hotel_id}: {len(room_pairs)} 个房型对，分为 {len(batches)} 个批次")

                for batch_index, batch_pairs in enumerate(batches):
                    all_tasks.append(
                        (hotel_id, batch_pairs, batch_index, room_pairs, data['country']))

            logger.info(f"总计 {len(all_tasks)} 个批次任务")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有批次任务
                future_to_task = {
                    executor.submit(
                        self.call_llm_for_room_pairs,
                        hotel_id,
                        batch_pairs,
                        batch_index
                    ): (hotel_id, batch_pairs, batch_index, all_room_pairs, country_code)
                    for hotel_id, batch_pairs, batch_index, all_room_pairs, country_code in all_tasks
                }

                # 收集结果按酒店分组
                hotel_batch_results = {}
                processed_batches = 0

                for future in as_completed(future_to_task):
                    hotel_id, batch_pairs, batch_index, all_room_pairs, country_code = future_to_task[
                        future]
                    try:
                        hotel_id_result, llm_response, batch_index_result = future.result()

                        if hotel_id_result not in hotel_batch_results:
                            hotel_batch_results[hotel_id_result] = {
                                'batches': {},
                                'all_room_pairs': all_room_pairs,
                                'country_code': country_code
                            }

                        # 解析批次结果
                        if not llm_response.startswith("ERROR:"):
                            comparison_results = self.parse_comparison_results(
                                llm_response, batch_pairs)
                            hotel_batch_results[hotel_id_result]['batches'][batch_index_result] = comparison_results

                            # 立即写入批次结果
                            if comparison_results:
                                self.save_single_hotel_results(
                                    hotel_id_result,
                                    country_code,
                                    comparison_results,
                                    self.output_file
                                )

                            logger.info(
                                f"酒店 {hotel_id_result} 批次 {batch_index_result} 完成并写入: {len(comparison_results)}/{len(batch_pairs)} 个比较")
                        else:
                            hotel_batch_results[hotel_id_result]['batches'][batch_index_result] = [
                            ]
                            logger.error(
                                f"酒店 {hotel_id_result} 批次 {batch_index_result} LLM调用失败: {llm_response}")

                        processed_batches += 1
                        progress = (processed_batches / len(all_tasks)) * 100
                        logger.info(
                            f"批次进度: {processed_batches}/{len(all_tasks)} ({progress:.1f}%)")

                    except Exception as e:
                        logger.error(
                            f"处理批次失败 (hotel_id {hotel_id}, batch {batch_index}): {str(e)}")
                        if hotel_id not in hotel_batch_results:
                            hotel_batch_results[hotel_id] = {
                                'batches': {},
                                'all_room_pairs': all_room_pairs,
                                'country_code': country_code
                            }
                        hotel_batch_results[hotel_id]['batches'][batch_index] = [
                        ]
                        processed_batches += 1

            # 合并每个酒店的所有批次结果并进行校验
            for hotel_id, batch_data in hotel_batch_results.items():
                all_comparisons = []
                for batch_index in sorted(batch_data['batches'].keys()):
                    batch_results = batch_data['batches'][batch_index]
                    all_comparisons.extend(batch_results)

                results[hotel_id] = all_comparisons
                room_list = hotel_room_data[hotel_id]['rooms']

                # 进行数量校验
                validation_result = self.validate_comparison_count(
                    hotel_id,
                    room_list,
                    all_comparisons
                )
                all_validations.append(validation_result)

                processed_hotels += 1
                logger.info(
                    f"酒店 {hotel_id} 所有批次合并完成: {len(all_comparisons)} 个比较结果")

            # 处理没有结果的酒店（如果有的话）
            for hotel_id in hotel_room_data.keys():
                if hotel_id not in results:
                    results[hotel_id] = []

                    # 写入错误记录
                    self.save_single_hotel_results(
                        hotel_id,
                        hotel_room_data[hotel_id]['country'],
                        [],  # 空的比较结果
                        self.output_file
                    )

                    validation_result = {
                        'hotel_id': hotel_id,
                        'room_count': len(hotel_room_data[hotel_id]['rooms']),
                        'expected_comparisons': self.calculate_expected_comparisons(len(hotel_room_data[hotel_id]['rooms'])),
                        'actual_comparisons': 0,
                        'is_complete': False,
                        'completion_rate': 0,
                        'result_stats': {'NO_RESULT': 1},
                        'missing_pairs_count': 0,
                        'missing_pairs': []
                    }
                    all_validations.append(validation_result)

            logger.info("所有酒店房型排序处理完成")
            logger.info(f"📁 结果已实时写入到: {self.output_file}")

            # 输出整体校验摘要
            self.log_overall_validation_summary(all_validations)

            return results, all_validations

        except Exception as e:
            logger.error(f"处理房型排序失败: {str(e)}")
            raise

    def save_results(self, hotel_room_data: Dict[int, Dict], ranking_results: Dict[int, List[Dict]],
                     validation_results: List[Dict], output_file: str):
        """保存排序结果"""
        try:
            results_data = []
            validation_lookup = {v['hotel_id']: v for v in validation_results}

            for hotel_id, room_data in hotel_room_data.items():
                comparison_results = ranking_results.get(hotel_id, [])
                validation_info = validation_lookup.get(hotel_id, {})

                if comparison_results:
                    # 为每个比较结果创建一行数据
                    for comparison in comparison_results:
                        results_data.append({
                            's_hotel_id': hotel_id,
                            'country_code': room_data['country'],
                            'room_a': comparison['room_a'],
                            'room_b': comparison['room_b'],
                            'comparison_result': comparison['comparison_result'],
                            'reason': comparison['reason']
                        })
                else:
                    # 如果没有比较结果，记录错误
                    results_data.append({
                        's_hotel_id': hotel_id,
                        'country_code': room_data['country'],
                        'room_a': '',
                        'room_b': '',
                        'comparison_result': 'ERROR',
                        'reason': '处理失败或无比较结果'
                    })

            results_df = pd.DataFrame(results_data)
            results_df.to_csv(output_file, index=False, encoding='utf-8')

            logger.info(f"结果已保存到: {output_file}")

            # 显示统计信息
            total_comparisons = len(results_df)
            error_count = len(
                results_df[results_df['comparison_result'] == 'ERROR'])
            success_count = total_comparisons - error_count

            logger.info(f"总比较次数: {total_comparisons}")
            logger.info(
                f"成功比较: {success_count} ({success_count/total_comparisons*100:.1f}%)")
            logger.info(
                f"失败比较: {error_count} ({error_count/total_comparisons*100:.1f}%)")

            # 显示比较结果分布
            if success_count > 0:
                result_stats = results_df[results_df['comparison_result']
                                          != 'ERROR']['comparison_result'].value_counts()
                logger.info("比较结果分布:")
                for result, count in result_stats.items():
                    logger.info(f"  {result}: {count}")

            # 显示国家分布
            country_stats = results_df['country_code'].value_counts()
            logger.info("国家分布:")
            for country, count in country_stats.head(10).items():
                logger.info(f"  {country}: {count}")

        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
            raise

    def run_ranking_analysis(self, csv_file_path: str, target_hotel_count: int = 1000,
                             max_workers: int = 5, batch_size: int = 20, hotel_ids: List[int] = None) -> str:
        """运行房型排序分析

        Args:
            csv_file_path: CSV文件路径
            target_hotel_count: 目标采样酒店数量（当hotel_ids为None时使用）
            max_workers: 最大并发线程数
            batch_size: 每批处理的房型对数量
            hotel_ids: 指定要处理的酒店ID列表，如果提供则直接处理这些酒店

        Returns:
            输出文件路径
        """
        try:
            # 1. 加载和采样数据
            logger.info("第1步：加载和采样数据")
            sampled_df = self.load_and_sample_hotels(
                csv_file_path, target_hotel_count, hotel_ids)

            # 2. 准备房型数据
            logger.info("第2步：准备房型数据")
            hotel_room_data = self.prepare_hotel_room_data(sampled_df)

            if not hotel_room_data:
                logger.warning("没有找到有多个房型的酒店")
                return None

            # 输出期望比较数量的预览
            logger.info("📊 数量校验预览:")
            total_expected_comparisons = 0

            for hotel_id, data in hotel_room_data.items():
                room_count = len(data['rooms'])
                expected = self.calculate_expected_comparisons(room_count)
                total_expected_comparisons += expected

                logger.info(
                    f"  酒店 {hotel_id}: {room_count} 房型 -> 期望 {expected} 条比较")

            logger.info(f"🎯 总期望比较数量: {total_expected_comparisons}")

            # 3. 处理房型排序（结果会实时写入CSV）
            logger.info("第3步：处理房型排序（实时写入结果）")
            ranking_results, validation_results = self.process_hotel_rankings(
                hotel_room_data, max_workers, batch_size)

            # 输出文件已经在process_hotel_rankings中实时生成
            output_file = self.output_file

            logger.info("✅ 所有结果已实时写入完成")
            return output_file

        except Exception as e:
            logger.error(f"运行排序分析失败: {str(e)}")
            raise


def main():
    """主函数"""
    csv_file_path = "/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/s_room_with_country_2025-07-21_095100.csv"

    # 创建处理器
    processor = RoomRankingProcessor()

    print("开始房型排序分析...")
    print(f"目标文件: {csv_file_path}")

    # 获取配置
    hotel_ids = BASE_CONFIG['hotel_ids']
    target_hotel_count = BASE_CONFIG['target_hotel_count']
    max_workers = BASE_CONFIG['max_workers']
    batch_size = BASE_CONFIG['batch_size']

    if hotel_ids:
        print(f"处理指定的 {len(hotel_ids)} 个酒店")
        for hotel_id in hotel_ids[:5]:  # 只显示前5个
            print(f"  - 酒店ID: {hotel_id}")
        if len(hotel_ids) > 5:
            print(f"  ... 等共 {len(hotel_ids)} 个酒店")
    else:
        print(f"采样 {target_hotel_count} 个酒店，按国家比例分布")

    print(f"使用 {max_workers} 个线程并发处理，每批次最多 {batch_size} 个房型对比")

    try:
        output_file = processor.run_ranking_analysis(
            csv_file_path=csv_file_path,
            target_hotel_count=target_hotel_count,
            max_workers=max_workers,
            batch_size=batch_size,
            hotel_ids=hotel_ids
        )

        if output_file:
            print(f"\n✅ 处理完成！")
            print(f"📁 输出文件: {output_file}")

            # 显示一些示例结果
            results_df = pd.read_csv(output_file)
            print(f"\n📊 处理统计:")
            print(f"总比较记录: {len(results_df)}")

            # 统计成功的比较
            success_df = results_df[results_df['comparison_result'] != 'ERROR']
            if len(success_df) > 0:
                print(f"成功比较: {len(success_df)}")
                print(f"比较结果分布:")
                for result, count in success_df['comparison_result'].value_counts().items():
                    print(f"  {result}: {count}")

            # 显示处理统计
            print(f"\n📊 处理统计:")
            total_comparisons = len(results_df)
            error_count = len(
                results_df[results_df['comparison_result'] == 'ERROR'])
            success_count = total_comparisons - error_count

            print(f"总比较记录: {total_comparisons}")
            print(f"成功比较: {success_count}")
            print(f"失败比较: {error_count}")

            # 按酒店统计
            hotel_count = results_df['s_hotel_id'].nunique()
            print(f"处理酒店数: {hotel_count}")

            print(f"\n🏨 示例结果:")
            # 显示一些成功的比较示例
            sample_df = success_df.head(5) if len(
                success_df) > 0 else results_df.head(5)
            for i, row in sample_df.iterrows():
                hotel_id = row['s_hotel_id']
                country = row['country_code']
                print(f"酒店ID: {hotel_id} ({country})")

                if row['comparison_result'] != 'ERROR':
                    print(f"比较: {row['room_a']} vs {row['room_b']}")
                    print(
                        f"结果: {row['comparison_result']} - {row['reason'][:100]}...")
                else:
                    print(f"处理失败: {row['reason']}")
                print("-" * 50)

        else:
            print("❌ 处理失败，请检查日志")

    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        logger.error(f"主程序执行失败: {str(e)}")


if __name__ == "__main__":
    main()
