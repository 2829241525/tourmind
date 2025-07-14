#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房间分组处理器 v2 - 优化版
读取CSV文件，获取similarity_score在0.9分以下的数据
调用LLM判断spl_room_text和s_room_text是否能分组

特点：
- 基于真实客户感知能力判断
- 区分明显差异和微小差异  
- 特殊处理笼统描述的降级风险
- 支持批量处理，提高效率

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

# 优先使用本地langchain源码
LANGCHAIN_MASTER_PATH = "/home/maxon/disk2/roomMatch/room_match/Langchain/langchain-master"
sys.path.insert(0, os.path.join(LANGCHAIN_MASTER_PATH, "libs", "langchain"))
sys.path.insert(0, os.path.join(LANGCHAIN_MASTER_PATH, "libs", "core"))
sys.path.insert(0, os.path.join(
    LANGCHAIN_MASTER_PATH, "libs", "partners", "openai"))

# 导入langchain组件


# LLM配置
LLM_CONFIG = {
    'qwen': {
        'model_name': 'qwen-plus',
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

判断原则（基于客户真实感知能力）：

**客户能够明显感知的差异（会投诉）**：
- 床数量明显差异：1张床 vs 2张床
- 床型实质差异：单人床 vs 双人床 vs 大床（不包括同类型的尺寸微差）
- 房间等级明显差异：standard vs deluxe vs suite
- 关键设施差异：有厨房 vs 无厨房，有阳台 vs 无阳台
- 景观实质差异：海景 vs 山景 vs 无景观

**客户无法感知或不会投诉的差异（可以分组）**：
- 重复描述差异：如"2 double beds,2 double beds" vs "2 double beds"
- 表述顺序差异：如"standard triple room" vs "triple triple standard"
- 语言差异：如"双人房" vs "double room"
- 描述详细程度差异：如"room,2 double beds" vs "2 double beds"（实质都是双床房）
- 床型尺寸微小差别：queen vs full（客户往往无法精确区分）
- 房间面积的小幅差异
- 装修时间等附加信息：newly renovated vs 普通

**笼统描述的特殊处理**：
- 如果一方描述笼统但另一方明确，需评估是否存在实质降级
- 例如："superior double/twin" vs "superior room" - 前者有床型选择权，后者无
- 但"room,2 double beds" vs "2 double beds" - 实质都提供2张双人床

房型组合数据：
{room_pairs}

请按照以下格式返回结果，每组一行：
组1: NO  | 作为客户我看到的房型描述中的房型/床型/景观为：xxx，实际入住发现房型/床型/景观为xxx，他们的差异让我受到了影响。
组2: YES | 不投诉
...

分析要求：
- 如果判断为NO，请明确说明客户能够明显感知到的实质性差异。只有判断为大概率会投诉的情况，才给出NO，否则给出YES。
- 如果判断为YES，无需说明详细原因
- 重点关注客户入住时能够直观察觉的实质差异，忽略表述方式、重复描述等非实质性差异
- 对于重复描述、表述顺序差异、同义词差异等，应判断为YES"""
        else:
            self.prompt_template = """你现在是一位普通酒店客户，请从真实客户体验的角度判断以下房型组合是否可以分为同一组。

请对每组房型进行判断，模拟两种入住场景是否会产生客户投诉：
- 场景1: 客户预订时看到房型A，实际入住时提供房型B
- 场景2: 客户预订时看到房型B，实际入住时提供房型A

分析步骤：
1. **首先消除重复描述和冗余信息**：忽略重复词汇（如"2 double beds,2 double beds"视为"2 double beds"）
2. **提取核心信息**：房间等级、床型、床数量、关键设施、景观等
3. **按最低配置理解模糊描述**：对于不明确的描述，按最基础配置处理
4. **评估or选择的风险**：描述中有"or"表示可能提供任意一种，需考虑最差情况

判断原则（基于客户真实感知能力）：

**客户能够明显感知的差异（会投诉）**：
- 床数量明显差异：1张床 vs 2张床
- 床型实质差异：单人床 vs 双人床 vs 大床（不包括同类型的尺寸微差）
- 房间等级明显差异：standard vs deluxe vs suite
- 关键设施差异：有厨房 vs 无厨房，有阳台 vs 无阳台
- 景观实质差异：海景 vs 山景 vs 无景观

**客户无法感知或不会投诉的差异（可以分组）**：
- 重复描述差异：如"2 double beds,2 double beds" vs "2 double beds"
- 表述顺序差异：如"standard triple room" vs "triple triple standard"
- 语言差异：如"双人房" vs "double room"
- 描述详细程度差异：如"room,2 double beds" vs "2 double beds"（实质都是双床房）
- 床型尺寸微小差别：queen vs full（客户往往无法精确区分）
- 房间面积的小幅差异
- 装修时间等附加信息：newly renovated vs 普通

**笼统描述的特殊处理**：
- 如果一方描述笼统但另一方明确，需评估是否存在实质降级
- 例如："superior double/twin" vs "superior room" - 前者有床型选择权，后者无
- 但"room,2 double beds" vs "2 double beds" - 实质都提供2张双人床

房型组合数据：
{room_pairs}

请按照以下格式返回结果，每组一行：
组1: YES/NO
组2: YES/NO
组3: YES/NO
...

只需要返回YES（可以分组，不会产生投诉）或NO（不可以分组，可能产生投诉），重点关注客户入住时能够直观察觉的实质差异，忽略表述方式、重复描述等非实质性差异。对于重复描述、表述顺序差异、同义词差异等，应判断为YES。"""

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
        """解析批量LLM返回结果，返回(判断结果列表, 分析原因列表)"""
        try:
            decisions = []
            analyses = []
            lines = llm_response.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if self.include_analysis:
                    # 解析包含分析的格式: "组X: YES/NO | 分析原因"
                    if ':' in line:
                        decision_part = line.split(':', 1)[1].strip()

                        if '|' in decision_part:
                            parts = decision_part.split('|', 1)
                            judgment = parts[0].strip().upper()
                            analysis = parts[1].strip() if len(
                                parts) > 1 else ""
                        else:
                            judgment = decision_part.upper()
                            analysis = ""

                        if 'YES' in judgment:
                            decisions.append('YES')
                        elif 'NO' in judgment:
                            decisions.append('NO')
                        else:
                            decisions.append('UNKNOWN')

                        analyses.append(analysis)

                    elif '|' in line:
                        parts = line.split('|', 1)
                        judgment = parts[0].strip().upper()
                        analysis = parts[1].strip() if len(parts) > 1 else ""

                        if 'YES' in judgment:
                            decisions.append('YES')
                        elif 'NO' in judgment:
                            decisions.append('NO')
                        else:
                            decisions.append('UNKNOWN')

                        analyses.append(analysis)

                    elif 'YES' in line.upper():
                        decisions.append('YES')
                        analyses.append("")
                    elif 'NO' in line.upper():
                        decisions.append('NO')
                        analyses.append("")

                else:
                    # 不包含分析的解析逻辑
                    if ':' in line:
                        decision_part = line.split(':', 1)[1].strip().upper()
                        if 'YES' in decision_part:
                            decisions.append('YES')
                        elif 'NO' in decision_part:
                            decisions.append('NO')
                        else:
                            decisions.append('UNKNOWN')
                    elif 'YES' in line.upper():
                        decisions.append('YES')
                    elif 'NO' in line.upper():
                        decisions.append('NO')

                    analyses.append("")

            # 确保结果数量匹配
            while len(decisions) < expected_count:
                decisions.append('UNKNOWN')
                analyses.append("")

            if len(decisions) > expected_count:
                decisions = decisions[:expected_count]
                analyses = analyses[:expected_count]

            logger.info(
                f"批量解析成功，预期{expected_count}个结果，实际获得{len(decisions)}个结果")
            return decisions, analyses

        except Exception as e:
            logger.error(f"批量结果解析失败: {str(e)}")
            return ['ERROR'] * expected_count, [''] * expected_count

    def call_llm_for_batch_grouping(self, room_pairs: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        """批量调用LLM判断房间是否能分组"""
        try:
            prompt = self._build_batch_prompt(room_pairs)
            messages = [HumanMessage(content=prompt)]
            result = self.llm.invoke(messages)

            llm_response = result.content if hasattr(
                result, 'content') else str(result)
            return self._parse_batch_results(llm_response, len(room_pairs))

        except Exception as e:
            logger.error(f"批量调用LLM失败: {str(e)}")
            return ['ERROR'] * len(room_pairs), [''] * len(room_pairs)

    def process_csv_file(self, csv_file_path: str, similarity_threshold: float = 0.9,
                         max_rows: int = 20, batch_size: int = 20) -> str:
        """处理CSV文件"""
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

            # 添加结果列
            filtered_df['llm_grouping_decision'] = ''
            if self.include_analysis:
                filtered_df['llm_grouping_analysis'] = ''

            # 批量处理
            total_count = len(filtered_df)
            processed_count = 0
            logger.info(f"开始批量处理，每批{batch_size}条数据...")

            for batch_start in range(0, total_count, batch_size):
                batch_end = min(batch_start + batch_size, total_count)
                batch_df = filtered_df.iloc[batch_start:batch_end]

                try:
                    # 准备批次数据
                    room_pairs = []
                    batch_indices = []

                    for index, row in batch_df.iterrows():
                        spl_room_text = str(row['spl_room_text'])
                        s_room_text = str(row['s_room_text'])
                        room_pairs.append((spl_room_text, s_room_text))
                        batch_indices.append(index)

                    # 调用LLM
                    logger.info(f"处理批次 {batch_start+1}-{batch_end}...")
                    batch_decisions, batch_analyses = self.call_llm_for_batch_grouping(
                        room_pairs)

                    # 更新数据
                    for index, decision, analysis in zip(batch_indices, batch_decisions, batch_analyses):
                        filtered_df.at[index,
                                       'llm_grouping_decision'] = decision
                        if self.include_analysis:
                            filtered_df.at[index,
                                           'llm_grouping_analysis'] = analysis
                        processed_count += 1

                    logger.info(
                        f"批次处理完成: {processed_count}/{total_count} ({processed_count/total_count*100:.1f}%)")
                    time.sleep(1.0)  # 避免API限制

                except Exception as e:
                    logger.error(
                        f"处理批次 {batch_start+1}-{batch_end} 时出错: {str(e)}")
                    for index in batch_indices:
                        filtered_df.at[index,
                                       'llm_grouping_decision'] = 'ERROR'
                        if self.include_analysis:
                            filtered_df.at[index, 'llm_grouping_analysis'] = ''
                        processed_count += 1

            # 保存结果
            output_file = csv_file_path.replace('.csv', '_with_grouping.csv')
            filtered_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"处理完成，结果已保存到: {output_file}")

            # 统计结果
            yes_count = len(
                filtered_df[filtered_df['llm_grouping_decision'] == 'YES'])
            no_count = len(
                filtered_df[filtered_df['llm_grouping_decision'] == 'NO'])
            error_count = len(
                filtered_df[filtered_df['llm_grouping_decision'] == 'ERROR'])
            unknown_count = len(
                filtered_df[filtered_df['llm_grouping_decision'] == 'UNKNOWN'])

            logger.info("分组结果统计:")
            logger.info(f"  YES (可以分组): {yes_count}")
            logger.info(f"  NO (不可以分组): {no_count}")
            logger.info(f"  ERROR (处理错误): {error_count}")
            logger.info(f"  UNKNOWN (未知): {unknown_count}")

            return output_file

        except Exception as e:
            logger.error(f"处理CSV文件失败: {str(e)}")
            raise


def main():
    """主函数"""
    csv_file_path = "/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/room_group_train_fix_clean_filtered_set_different_with_similarity.csv"

    # 创建处理器（默认包含分析）
    processor = RoomGroupingProcessor(include_analysis=True)

    print("开始处理20条数据（包含分析原因）...")
    try:
        output_file = processor.process_csv_file(
            csv_file_path,
            similarity_threshold=0.9,
            max_rows=20,
            batch_size=10
        )

        print(f"处理完成，输出文件: {output_file}")

        # 显示结果示例
        print("\n=== 分析结果示例 ===")
        df = pd.read_csv(output_file)
        for i, row in df.head(5).iterrows():
            print(f"数据{i+1}:")
            print(f"  房型A: {row['spl_room_text']}")
            print(f"  房型B: {row['s_room_text']}")
            print(f"  判断: {row['llm_grouping_decision']}")
            print(f"  分析: {row['llm_grouping_analysis']}")
            print()

    except Exception as e:
        print(f"处理失败: {str(e)}")


if __name__ == "__main__":
    main()
