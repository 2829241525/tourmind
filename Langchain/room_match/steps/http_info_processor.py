#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤1：HTTP信息处理器
访问HTTP请求，调用大模型生成房型匹配规则
"""

from dl_tool import create_dl_matcher, create_room_matching_tool
from login_handler import get_login_handler
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from prompts import get_prompt
from config import LLM_CONFIG, ROOM_MATCHING_CONFIG
import requests
import json
import logging
import sys
import os
import random
import re
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

# 添加父目录到路径以支持直接运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 临时添加父目录到路径以导入正确的配置
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


logger = logging.getLogger(__name__)


class HttpInfoProcessor:
    """HTTP信息处理器 - 步骤1"""

    # 默认API配置
    DEFAULT_API_CONFIG = {
        "base_url": "https://erp.tourmind.cn/roomtypemapping/api/get_roomtype",
        "default_params": {
            "supplierID": "34",
            "sRoomBaseType": "tourmind",
            "sroom_status": "0"
        },
        "default_headers": {
        }
    }

    def __init__(self, llm_type='qwen', enable_dl_matching=True, room_matching_config=None, **kwargs):
        """初始化HTTP信息处理器"""
        self.llm_type = llm_type
        self.llm_config = LLM_CONFIG.get(llm_type, LLM_CONFIG['qwen'])
        self.config = kwargs
        self.enable_dl_matching = enable_dl_matching
        self.room_matching_config = room_matching_config or ROOM_MATCHING_CONFIG

        # 初始化登录处理器
        self.login_handler = get_login_handler()

        # 初始化大模型
        self._setup_llm()

        # 初始化DL匹配器
        if self.enable_dl_matching:
            try:
                self.dl_matcher = create_dl_matcher()
                logger.info("DL房型匹配器初始化成功")
            except Exception as e:
                logger.warning(f"DL房型匹配器初始化失败: {str(e)}")
                self.dl_matcher = None
        else:
            self.dl_matcher = None

    def normalize_spaces(self, text: str) -> str:
        """将字符串中的多个连续空格替换为一个空格"""
        if not text:
            return text
        # 使用正则表达式将多个连续空格（包括制表符、换行符等）替换为一个空格
        return re.sub(r'\s+', ' ', text.strip())

    def parse_llm_matching_result(self, llm_response: str, hotel_id: str) -> Dict[str, List[Dict[str, str]]]:
        """解析大模型的匹配结果

        Args:
            llm_response: 大模型的输出结果
            hotel_id: 酒店ID

        Returns:
            包含匹配结果的字典，格式：
            {
                'successful_matches': [{'hotel_id', 'supplier_room', 'standard_room', 'reasoning'}],
                'unsuccessful_matches': [{'hotel_id', 'supplier_room', 'reason_category', 'reason_detail'}]
            }
        """
        try:
            successful_matches = []
            unsuccessful_matches = []

            # 尝试提取JSON代码块
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, llm_response, re.DOTALL)

            if json_match:
                json_content = json_match.group(1)
                logger.info("从Markdown代码块中提取到JSON内容")
            else:
                # 如果没有找到代码块，尝试查找JSON对象
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}')

                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_content = llm_response[json_start:json_end + 1]
                    logger.info("从文本中提取到JSON内容")
                else:
                    json_content = llm_response.strip()
                    logger.info("使用完整响应作为JSON内容")

            # 尝试解析JSON格式的结果
            try:
                parsed_result = json.loads(json_content)
                logger.info("成功解析JSON格式")

                # 处理新的JSON结构
                if isinstance(parsed_result, dict):
                    # 处理成功匹配的结果
                    successful_list = parsed_result.get(
                        'successful_matches', [])
                    if isinstance(successful_list, list):
                        for match in successful_list:
                            if isinstance(match, dict):
                                supplier_room = match.get('supplier_room', '')
                                standard_room = match.get(
                                    'matched_standard_room', '')
                                reasoning = match.get('reasoning', '')

                                if supplier_room and standard_room:
                                    successful_matches.append({
                                        'hotel_id': hotel_id,
                                        'supplier_room': supplier_room,
                                        'standard_room': standard_room,
                                        'reasoning': reasoning,
                                        'match_status': 'successful'
                                    })

                    # 处理无法匹配的结果
                    unsuccessful_list = parsed_result.get(
                        'unsuccessful_matches', [])
                    if isinstance(unsuccessful_list, list):
                        for match in unsuccessful_list:
                            if isinstance(match, dict):
                                supplier_room = match.get('supplier_room', '')
                                reason_category = match.get(
                                    'reason_category', '')
                                reason_detail = match.get('reason_detail', '')

                                if supplier_room:
                                    unsuccessful_matches.append({
                                        'hotel_id': hotel_id,
                                        'supplier_room': supplier_room,
                                        'standard_room': '',  # 无匹配
                                        'reasoning': '',
                                        'reason_category': reason_category,
                                        'reason_detail': reason_detail,
                                        'match_status': 'unsuccessful'
                                    })

                logger.info(
                    f"成功解析JSON格式的匹配结果：成功匹配{len(successful_matches)}条，无法匹配{len(unsuccessful_matches)}条")

                return {
                    'successful_matches': successful_matches,
                    'unsuccessful_matches': unsuccessful_matches
                }

            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析失败: {str(e)}，尝试文本解析")

            # 降级到文本格式解析（保持向后兼容）
            lines = llm_response.strip().split('\n')

            # 寻找常见的匹配格式
            patterns = [
                # 格式: 供应商房型 -> 标准房型
                r'(.+?)\s*(?:->|→|匹配到|对应|匹配标准房型)\s*[:：]?\s*(.+)',
                # 格式: [1]: 供应商房型 -> 标准房型
                r'\[\d+\]:\s*(.+?)\s*(?:->|→|匹配到|对应|匹配标准房型)\s*[:：]?\s*(.+)',
                # 格式: 1. 供应商房型 -> 标准房型
                r'\d+\.\s*(.+?)\s*(?:->|→|匹配到|对应|匹配标准房型)\s*[:：]?\s*(.+)',
                # 格式: "供应商房型" : "标准房型"
                r'["\'](.+?)["\']?\s*[:：]\s*["\'](.+?)["\']?',
                # 格式: 供应商房型 | 标准房型
                r'(.+?)\s*\|\s*(.+)',
            ]

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//') or line.startswith('-'):
                    continue

                # 检查是否是无法匹配的行
                if '无法匹配' in line or 'unsuccessful' in line.lower() or '不匹配' in line:
                    # 尝试提取供应商房型名称
                    unmatch_patterns = [
                        r'供应商房型[:：]\s*(.+?)\s*(?:->|→)\s*无法匹配',
                        r'(.+?)\s*(?:->|→)\s*无法匹配',
                        r'供应商房型[:：]\s*(.+)',
                    ]

                    for pattern in unmatch_patterns:
                        match = re.search(pattern, line)
                        if match:
                            supplier_room = match.group(1).strip().strip('"\'')
                            if supplier_room:
                                unsuccessful_matches.append({
                                    'hotel_id': hotel_id,
                                    'supplier_room': supplier_room,
                                    'standard_room': '',
                                    'reasoning': '',
                                    'reason_category': 'UNMATCHABLE',
                                    'reason_detail': '从文本解析获得，具体原因未详细说明',
                                    'match_status': 'unsuccessful'
                                })
                                break
                    continue

                # 尝试匹配成功的格式
                for pattern in patterns:
                    match = re.match(pattern, line)
                    if match:
                        supplier_room = match.group(1).strip()
                        standard_room = match.group(2).strip()

                        # 清理可能的前缀标记
                        supplier_room = re.sub(
                            r'^\[\d+\]:\s*', '', supplier_room).strip()
                        supplier_room = re.sub(
                            r'^\d+\.\s*', '', supplier_room).strip()
                        supplier_room = re.sub(
                            r'^供应商房型[:：]\s*', '', supplier_room).strip()
                        standard_room = re.sub(
                            r'^\[\d+\]:\s*', '', standard_room).strip()
                        standard_room = re.sub(
                            r'^\d+\.\s*', '', standard_room).strip()
                        standard_room = re.sub(
                            r'^匹配标准房型[:：]\s*', '', standard_room).strip()

                        # 去除引号
                        supplier_room = supplier_room.strip('"\'')
                        standard_room = standard_room.strip('"\'')

                        if supplier_room and standard_room and '无法匹配' not in standard_room:
                            successful_matches.append({
                                'hotel_id': hotel_id,
                                'supplier_room': supplier_room,
                                'standard_room': standard_room,
                                'reasoning': '从文本解析获得',
                                'match_status': 'successful'
                            })
                            break

            logger.info(
                f"文本解析完成：成功匹配{len(successful_matches)}条，无法匹配{len(unsuccessful_matches)}条")

            return {
                'successful_matches': successful_matches,
                'unsuccessful_matches': unsuccessful_matches
            }

        except Exception as e:
            logger.error(f"解析LLM匹配结果失败: {str(e)}")
            return {
                'successful_matches': [],
                'unsuccessful_matches': []
            }

    def save_matching_results_to_csv(self, matching_results: Dict[str, List[Dict[str, str]]], hotel_id: str) -> str:
        """将匹配结果保存到CSV文件

        Args:
            matching_results: 匹配结果字典，包含successful_matches和unsuccessful_matches
            hotel_id: 酒店ID

        Returns:
            保存的CSV文件路径
        """
        try:
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 确保output目录存在
            output_dir = os.path.join(
                os.path.dirname(__file__), '..', 'output')
            os.makedirs(output_dir, exist_ok=True)

            # 生成文件名
            csv_filename = f"llm_matching_results_{hotel_id}_{timestamp}.csv"
            csv_path = os.path.join(output_dir, csv_filename)

            # 准备CSV数据
            csv_data = []

            # 处理成功匹配的结果
            successful_matches = matching_results.get('successful_matches', [])
            for result in successful_matches:
                csv_data.append({
                    '酒店ID': result.get('hotel_id', hotel_id),
                    '供应商房型+床型': result.get('supplier_room', ''),
                    '标准房型+床型': result.get('standard_room', ''),
                    '匹配状态': '成功匹配',
                    '匹配理由': result.get('reasoning', ''),
                    '失败原因分类': '',
                    '失败原因详情': ''
                })

            # 处理无法匹配的结果
            unsuccessful_matches = matching_results.get(
                'unsuccessful_matches', [])
            for result in unsuccessful_matches:
                csv_data.append({
                    '酒店ID': result.get('hotel_id', hotel_id),
                    '供应商房型+床型': result.get('supplier_room', ''),
                    '标准房型+床型': '无法匹配',
                    '匹配状态': '无法匹配',
                    '匹配理由': '',
                    '失败原因分类': result.get('reason_category', ''),
                    '失败原因详情': result.get('reason_detail', '')
                })

            # 如果没有任何结果，创建一个空的CSV文件
            if not csv_data:
                csv_data.append({
                    '酒店ID': hotel_id,
                    '供应商房型+床型': '',
                    '标准房型+床型': '',
                    '匹配状态': '',
                    '匹配理由': '',
                    '失败原因分类': '',
                    '失败原因详情': ''
                })
                logger.warning("没有找到匹配结果，创建空的CSV文件")

            # 保存到CSV文件
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            total_successful = len(successful_matches)
            total_unsuccessful = len(unsuccessful_matches)
            total_results = total_successful + total_unsuccessful

            logger.info(f"匹配结果已保存到CSV文件: {csv_path}")
            logger.info(
                f"共保存{total_results}条结果：成功匹配{total_successful}条，无法匹配{total_unsuccessful}条")

            return csv_path

        except Exception as e:
            logger.error(f"保存CSV文件失败: {str(e)}")
            return ""

    def _setup_llm(self):
        """设置大模型客户端"""
        # 设置环境变量
        os.environ['OPENAI_API_KEY'] = self.llm_config['api_key']
        os.environ['DASHSCOPE_API_KEY'] = self.llm_config['api_key']

        self.llm = ChatOpenAI(
            model=self.llm_config['model_name'],
            temperature=self.llm_config['temperature'],
            max_tokens=self.llm_config['max_tokens'],
            api_key=self.llm_config['api_key'],
            base_url=self.llm_config['base_url'],
            streaming=True  # 启用流式输出
        )

        logger.info(f"大模型初始化成功: {self.llm_config['model_name']}")

    def build_api_url(self, hotel_id: str, **override_params) -> tuple:
        """构建API URL和请求配置

        Args:
            hotel_id: 酒店ID
            **override_params: 覆盖默认参数

        Returns:
            tuple: (url, headers)
        """
        # 合并参数
        params = self.DEFAULT_API_CONFIG["default_params"].copy()
        params["sHotelID"] = hotel_id
        params.update(override_params)

        # 构建URL
        base_url = self.DEFAULT_API_CONFIG["base_url"]
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{base_url}?{param_string}"

        # 更新Referer中的酒店ID
        headers = self.DEFAULT_API_CONFIG["default_headers"].copy()
        headers['Referer'] = f'https://erp.tourmind.cn/roomtypemapping/{hotel_id}?supplierId=0&brt=tourmind&loadMaster=true'

        return full_url, headers

    def process_http_request(self, url: str, method: str = "GET", headers: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """执行HTTP请求并获取响应"""
        try:
            default_headers = {
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'zh-CN,zh;q=0.9',
                'Connection': 'keep-alive',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }

            if headers:
                default_headers.update(headers)

            response = requests.request(
                method=method, url=url, headers=default_headers, verify=False, timeout=300, **kwargs)

            response_data = {
                "status_code": response.status_code,
                "response_data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                "metadata": {"url": url, "method": method, "success": response.status_code == 200}
            }

            logger.info(f"HTTP请求完成: {url}, 状态码: {response.status_code}")
            return response_data

        except Exception as e:
            logger.error(f"HTTP请求失败: {str(e)}")
            return {"error": str(e)}

    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的Jaccard相似度"""
        if not text1 or not text2:
            return 0.0

        # 转换为小写并分割为词集合
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # 计算交集和并集
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        # 计算Jaccard相似度
        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)

    def get_matched_examples_by_similarity(self, matched_pairs: List[Dict]) -> List[str]:
        """使用Jaccard相似度获取匹配案例"""
        try:
            if not matched_pairs:
                return []

            logger.info(f"使用Jaccard相似度计算，共{len(matched_pairs)}个匹配对")

            # 计算每个匹配对的Jaccard相似度
            similarity_data = []
            seen_pairs = set()  # 用于去重

            for i, pair in enumerate(matched_pairs):
                supplier_name = pair['supplier_room']['combined_name']
                standard_name = pair['standard_room']['combined_name']

                # 空格规范化处理
                normalized_supplier = self.normalize_spaces(supplier_name)
                normalized_standard = self.normalize_spaces(standard_name)

                # 创建唯一标识用于去重
                normalized_pair_key = f"{normalized_supplier} → {normalized_standard}"

                # 如果这个配对还没有见过，则计算相似度
                if normalized_pair_key not in seen_pairs:
                    seen_pairs.add(normalized_pair_key)

                    # 计算Jaccard相似度
                    jaccard_sim = self.calculate_jaccard_similarity(
                        normalized_supplier, normalized_standard)

                    similarity_data.append({
                        'pair_key': normalized_pair_key,
                        'standard_room': normalized_standard,
                        'jaccard_similarity': jaccard_sim,
                        'original_index': i
                    })

            # 按标准房型分组
            standard_room_groups = {}
            for item in similarity_data:
                standard_room = item['standard_room']
                if standard_room not in standard_room_groups:
                    standard_room_groups[standard_room] = []
                standard_room_groups[standard_room].append(item)

            # 对每个标准房型组内的案例按相似度排序（从低到高）
            for standard_room in standard_room_groups:
                standard_room_groups[standard_room].sort(
                    key=lambda x: x['jaccard_similarity'])

            logger.info(
                f"去重后共{len(similarity_data)}个唯一配对，涵盖{len(standard_room_groups)}种标准房型")

            # 选择策略
            selected_examples = []

            if len(standard_room_groups) <= 20:
                # 标准房型种类不超过20个，每种选择相似度最低的1个
                for standard_room, items in standard_room_groups.items():
                    if items:
                        selected_examples.append(items[0])  # 选择相似度最低的

                # 如果不足20个，继续从各组中选择
                round_index = 1
                while len(selected_examples) < 20:
                    added_in_round = False
                    for standard_room, items in standard_room_groups.items():
                        if len(selected_examples) >= 20:
                            break
                        if round_index < len(items):
                            selected_examples.append(items[round_index])
                            added_in_round = True

                    if not added_in_round:  # 所有组都已经用完
                        break
                    round_index += 1
            else:
                # 标准房型种类超过20个，随机选择20种，每种选择相似度最低的1个
                selected_standard_rooms = random.sample(
                    list(standard_room_groups.keys()), 20)
                for standard_room in selected_standard_rooms:
                    items = standard_room_groups[standard_room]
                    if items:
                        selected_examples.append(items[0])  # 选择相似度最低的

            # 按相似度排序（从低到高）
            selected_examples.sort(key=lambda x: x['jaccard_similarity'])

            # 格式化输出
            formatted_examples = []
            for i, item in enumerate(selected_examples, 1):
                formatted_examples.append(f"[{i}]: {item['pair_key']}")

            avg_similarity = sum(item['jaccard_similarity'] for item in selected_examples) / len(
                selected_examples) if selected_examples else 0
            logger.info(
                f"使用Jaccard相似度选择{len(formatted_examples)}个案例，平均相似度: {avg_similarity:.3f}")

            return formatted_examples

        except Exception as e:
            logger.error(f"Jaccard相似度计算失败: {str(e)}")

            # 降级处理：随机选择
            logger.info("使用降级方案：随机选择20个匹配案例")

            # 按标准房型分组
            standard_room_groups = {}
            seen_pairs = set()  # 用于去重

            for pair in matched_pairs:
                supplier_name = pair['supplier_room']['combined_name']
                standard_name = pair['standard_room']['combined_name']

                # 空格规范化处理
                normalized_supplier = self.normalize_spaces(supplier_name)
                normalized_standard = self.normalize_spaces(standard_name)
                normalized_pair_key = f"{normalized_supplier} → {normalized_standard}"

                # 如果这个配对还没有见过，则添加到对应的标准房型组
                if normalized_pair_key not in seen_pairs:
                    seen_pairs.add(normalized_pair_key)

                    if normalized_standard not in standard_room_groups:
                        standard_room_groups[normalized_standard] = []
                    standard_room_groups[normalized_standard].append(
                        normalized_pair_key)

            # 随机打乱每个组内的案例
            for standard_room in standard_room_groups:
                random.shuffle(standard_room_groups[standard_room])

            # 从每个标准房型组中选择案例，确保多样性
            selected_examples = []

            # 第一轮：每个标准房型选择1个案例
            for standard_room, pairs in standard_room_groups.items():
                if len(selected_examples) < 20 and pairs:
                    selected_examples.append(pairs[0])

            # 第二轮：如果还没达到20个，继续从各组中选择
            round_index = 1
            while len(selected_examples) < 20:
                added_in_round = False
                for standard_room, pairs in standard_room_groups.items():
                    if len(selected_examples) >= 20:
                        break
                    if round_index < len(pairs):
                        selected_examples.append(pairs[round_index])
                        added_in_round = True

                if not added_in_round:  # 所有组都已经用完
                    break
                round_index += 1

            # 格式化输出
            formatted_examples = []
            for i, pair_key in enumerate(selected_examples, 1):
                formatted_examples.append(f"[{i}]: {pair_key}")

            return formatted_examples

    def extract_room_types(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """从API响应中提取房型数据"""
        try:
            logger.info(f"开始提取房型数据，响应数据类型: {type(response_data)}")
            logger.info(
                f"响应数据结构: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")

            # 获取实际的响应数据
            actual_response = response_data.get("response_data", {})
            logger.info(f"实际响应数据类型: {type(actual_response)}")

            # 如果响应数据是字符串，尝试解析为JSON
            if isinstance(actual_response, str):
                try:
                    actual_response = json.loads(actual_response)
                    logger.info("成功将字符串响应解析为JSON")
                except json.JSONDecodeError:
                    logger.error("响应数据不是有效的JSON格式")
                    return {"error": "响应数据不是有效的JSON格式"}

            # 确保是字典类型
            if not isinstance(actual_response, dict):
                logger.error(f"响应数据不是字典格式，而是: {type(actual_response)}")
                return {"error": f"响应数据格式错误: {type(actual_response)}"}

            api_data = actual_response.get("Data", {})
            logger.info(
                f"API数据结构: {list(api_data.keys()) if isinstance(api_data, dict) else 'Not a dict'}")

            spl_room_type_list = api_data.get("SPLRoomTypeList", [])
            s_room_type_list = api_data.get("SRoomTypeList", [])

            # 创建标准房型字典
            standard_rooms_dict = {}
            for room in s_room_type_list:
                room_id = room.get("SRoomID")
                if room_id:
                    room_name = room.get("RoomTypeName", "")
                    bed_desc = room.get("BedTypeDesc", "")
                    combined_name = f"{room_name} {bed_desc}".strip()
                    standard_rooms_dict[room_id] = {
                        "combined_name": combined_name, "room_id": room_id}

            # 提取未匹配和已匹配房型
            unmatched_supplier_rooms = []
            matched_pairs = []

            for room in spl_room_type_list:
                room_name = room.get("RoomTypeName", "")
                bed_desc = room.get("BedTypeDesc", "")
                combined_name = f"{room_name} {bed_desc}".strip()
                supplier_room_data = {"combined_name": combined_name}

                sroom_id = room.get("SRoomID")
                if sroom_id == 0:
                    if combined_name:
                        unmatched_supplier_rooms.append(supplier_room_data)
                else:
                    if sroom_id in standard_rooms_dict and combined_name:
                        matched_pairs.append({
                            "supplier_room": supplier_room_data,
                            "standard_room": standard_rooms_dict[sroom_id]
                        })

            # 提取所有标准房型
            standard_room_types = []
            for room in s_room_type_list:
                room_name = room.get("RoomTypeName", "")
                bed_desc = room.get("BedTypeDesc", "")
                combined_name = f"{room_name} {bed_desc}".strip()
                if combined_name:
                    standard_room_types.append(
                        {"combined_name": combined_name})

            extracted_data = {
                "hotel_info": {
                    "hotel_id": api_data.get("SHotelID"),
                    "hotel_name": api_data.get("HotelName"),
                    "hotel_name_en": api_data.get("HotelNameEN"),
                    "city": api_data.get("City")
                },
                "unmatched_supplier_rooms": unmatched_supplier_rooms,
                "standard_room_types": standard_room_types,
                "matched_pairs": matched_pairs
            }

            logger.info(
                f"房型数据提取完成: 未匹配{len(unmatched_supplier_rooms)}个, 标准房型{len(standard_room_types)}个, 已匹配{len(matched_pairs)}个")
            return extracted_data

        except Exception as e:
            logger.error(f"房型数据提取失败: {str(e)}")
            return {"error": str(e)}

    def format_room_types_for_prompt(self, extracted_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """将房型数据格式化为结构化格式"""
        try:
            # 1. 通过相似度接口获取匹配案例
            matched_pairs = extracted_data.get("matched_pairs", [])
            matched_examples = self.get_matched_examples_by_similarity(
                matched_pairs)

            # 2. 未匹配的供应商房型 (格式: [1]: 房型名称)
            unmatched_rooms = extracted_data.get(
                "unmatched_supplier_rooms", [])
            unmatched_list = []
            unmatched_room_names = []
            # 限制最大数额为20
            max_rooms = min(20, len(unmatched_rooms))
            for i, room in enumerate(unmatched_rooms[:max_rooms], 1):
                room_name = room['combined_name']
                unmatched_list.append(f"[{i}]: {room_name}")
                unmatched_room_names.append(room_name)

            # 3. 标准房型 (格式: [1]: 房型名称)
            standard_rooms = extracted_data.get("standard_room_types", [])
            standard_list = []
            standard_room_names = []
            for i, room in enumerate(standard_rooms, 1):
                room_name = room['combined_name']
                standard_list.append(f"[{i}]: {room_name}")
                standard_room_names.append(room_name)

            # 4. 如果启用了DL匹配，计算未匹配房型与标准房型的相似度
            dl_similarity_result = None
            if self.dl_matcher and unmatched_room_names and standard_room_names:
                try:
                    logger.info("开始使用DL匹配器计算房型相似度...")
                    dl_similarity_result = self.dl_matcher.calculate_similarities(
                        unmatched_rooms=unmatched_room_names,
                        standard_rooms=standard_room_names
                    )

                    if dl_similarity_result["status"] == "success":
                        # 保存相似度结果到CSV
                        csv_path = self.dl_matcher.save_to_csv()
                        logger.info(f"DL匹配完成，相似度结果已保存到: {csv_path}")

                        # 获取最佳匹配
                        best_matches = self.dl_matcher.get_best_matches(
                            threshold=0.7)
                        if best_matches:
                            logger.info(
                                f"发现{len(best_matches)}个高质量匹配（相似度≥0.7）:")
                            for match in best_matches[:5]:  # 只显示前5个
                                logger.info(
                                    f"  {match['supplier_room']} → {match['standard_room']} (相似度: {match['similarity']:.4f})")
                    else:
                        logger.warning(
                            f"DL匹配失败: {dl_similarity_result.get('error', 'Unknown error')}")

                except Exception as e:
                    logger.error(f"DL匹配过程中出现异常: {str(e)}")
                    dl_similarity_result = None

            result = {
                "matched_examples": matched_examples,
                "unmatched_rooms": unmatched_list,
                "standard_rooms": standard_list
            }

            # 如果有DL匹配结果，添加到返回数据中
            if dl_similarity_result:
                result["dl_similarity_result"] = dl_similarity_result

            return result

        except Exception as e:
            logger.error(f"房型格式化失败: {str(e)}")
            return {"matched_examples": [], "unmatched_rooms": [], "standard_rooms": []}

    def generate_extraction_rules_with_llm(self, formatted_data: Dict[str, List[str]]) -> Dict[str, str]:
        """使用大模型生成房型匹配规则"""
        try:
            # 获取提示词模板
            prompt_template_obj = get_prompt('http_info_extraction')

            # 构建完整的prompt内容
            formatted_prompt = prompt_template_obj.format(
                matched_examples="\n".join(
                    formatted_data.get("matched_examples", [])),
                unmatched_rooms="\n".join(
                    formatted_data.get("unmatched_rooms", [])),
                standard_rooms="\n".join(
                    formatted_data.get("standard_rooms", []))
            )

            # # 输出转换后的完整prompt
            # logger.info("转换后的完整Prompt:")
            # logger.info("="*80)
            # logger.info(formatted_prompt)
            # logger.info("="*80)

            # 使用新的RunnableSequence语法创建链
            chain = prompt_template_obj | self.llm

            # 调用大模型，添加180秒超时
            logger.info("正在调用大模型生成房型匹配规则（超时时间：180秒）...")

            result = chain.invoke({
                "matched_examples": "\n".join(formatted_data.get("matched_examples", [])),
                "unmatched_rooms": "\n".join(formatted_data.get("unmatched_rooms", [])),
                "standard_rooms": "\n".join(formatted_data.get("standard_rooms", []))
            })
            llm_response = result.content if hasattr(
                result, 'content') else str(result)

            return {
                "llm_response": llm_response,
                "formatted_prompt": formatted_prompt,
            }

        except Exception as e:
            logger.error(f"大模型调用失败: {str(e)}")
            raise

    def generate_room_matching_with_agent(self, formatted_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """使用Agent方式进行房型匹配

        Args:
            formatted_data: 格式化的房型数据

        Returns:
            包含匹配结果的字典
        """
        try:
            logger.info("开始使用Agent方式进行房型匹配...")

            # 创建房型匹配工具
            room_matching_tool = create_room_matching_tool()
            if not room_matching_tool:
                raise Exception("无法创建房型匹配工具")

            # 创建Agent内存
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            # 初始化Agent
            agent_executor = initialize_agent(
                tools=[room_matching_tool],
                llm=self.llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=self.room_matching_config.get('agent_verbose', True),
                max_iterations=self.room_matching_config.get(
                    'agent_max_iterations', 4),  # 从3增加到10
                handle_parsing_errors=self.room_matching_config.get(
                    'agent_handle_parsing_errors', True),
                early_stopping_method=self.room_matching_config.get(
                    'agent_early_stopping_method', 'generate')
            )

            # 获取Agent提示词模板
            agent_prompt_template = get_prompt('agent_room_matching')
            if not agent_prompt_template:
                raise Exception("无法获取agent_room_matching提示词模板")

            # 构建Agent查询
            agent_query = agent_prompt_template.format(
                matched_examples="\n".join(
                    formatted_data.get("matched_examples", [])),
                unmatched_rooms="\n".join(
                    formatted_data.get("unmatched_rooms", [])),
                standard_rooms="\n".join(
                    formatted_data.get("standard_rooms", []))
            )

            # logger.info("构建的Agent查询:")
            # logger.info("="*80)
            # logger.info(agent_query)
            # logger.info("="*80)

            # 执行Agent查询
            logger.info("正在执行Agent房型匹配...")
            try:
                # 使用新的invoke方法替代已弃用的run方法
                response = agent_executor.invoke({"input": agent_query})
                agent_result = response.get("output", "")
            except Exception as invoke_error:
                logger.warning(f"invoke方法调用失败，尝试使用run方法: {str(invoke_error)}")
                # 如果invoke失败，回退到run方法（向后兼容）
                agent_result = agent_executor.run(agent_query)

            # 构建返回结果
            result = {
                "agent_response": agent_result,
                "agent_query": agent_query,
                "status": "success",
                "method": "agent_with_room_matching_tool"
            }

            logger.info("Agent房型匹配完成")
            return result

        except Exception as e:
            error_msg = f"Agent房型匹配失败: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "method": "agent_with_room_matching_tool"
            }

    def process_room_matching(self, formatted_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """根据配置选择房型匹配方式

        Args:
            formatted_data: 格式化的房型数据

        Returns:
            包含匹配结果的字典
        """
        matching_method = self.room_matching_config.get(
            'matching_method', 'agent')

        try:
            if matching_method == 'llm':
                logger.info("使用LLM直接调用方式进行房型匹配")
                result = self.generate_extraction_rules_with_llm(
                    formatted_data)
                # 标准化返回格式
                return {
                    "status": "success",
                    "method": "direct_llm_call",
                    "llm_response": result.get("llm_response", ""),
                    "formatted_prompt": result.get("formatted_prompt", ""),
                    "elapsed_time": result.get("elapsed_time", 0)
                }
            elif matching_method == 'agent':
                logger.info("使用Agent方式进行房型匹配")
                result = self.generate_room_matching_with_agent(formatted_data)
                return result
            else:
                error_msg = f"不支持的匹配方式: {matching_method}，支持的方式: 'llm', 'agent'"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "method": "unknown"
                }

        except Exception as e:
            error_msg = f"房型匹配处理失败: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "method": matching_method
            }

    def execute_step(self, hotel_id: str = None, url: str = None, method: str = "GET",
                     headers: Optional[Dict] = None, cookies: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """执行完整的步骤1流程

        Args:
            hotel_id: 酒店ID，优先使用此参数构建API请求
            url: 自定义URL，当hotel_id为None时使用
            method: HTTP方法
            headers: 自定义请求头
            cookies: 自定义cookies
            **kwargs: 其他参数
        """
        try:
            # 优先使用hotel_id构建API请求
            if hotel_id:
                api_url, api_headers = self.build_api_url(hotel_id, **kwargs)
                # 使用API配置，但允许用户覆盖
                final_url = api_url
                final_headers = api_headers

                # 用户提供的headers优先级更高
                if headers:
                    final_headers.update(headers)

                logger.info(f"使用酒店ID {hotel_id} 构建API请求")
            else:
                # 使用用户提供的URL
                if not url:
                    raise ValueError("必须提供hotel_id或url参数")
                final_url = url
                final_headers = headers or {}
                logger.info("使用自定义URL")

            # 获取cookies（优先级：用户提供 > 登录处理器存储）
            current_cookies = cookies or self.login_handler.get_current_cookies()

            # 处理cookies
            if current_cookies:
                if not final_headers:
                    final_headers = {}
                final_headers['Cookie'] = current_cookies
                logger.info(f"使用cookies: {current_cookies[:50]}...")
            else:
                logger.warning("未找到可用的cookies，可能需要先登录")

            # 1. 首先使用默认cookies执行HTTP请求
            response_data = self.process_http_request(
                url=final_url, method=method, headers=final_headers, **kwargs)

            if "error" in response_data:
                return {"step": 1, "status": "error", "error": response_data["error"]}

            # 检查是否是502错误，如果是则尝试重新登录
            if response_data.get("status_code") == 502 or "管理后台登录" in response_data.get("response_data"):
                logger.warning("检测到502错误，开始执行登录重试...")

                # 执行登录获取新cookies
                new_cookies = self.login_handler.login_and_get_cookies()

                if new_cookies:
                    logger.info(f"登录成功，cookies已自动保存: {new_cookies[:50]}...")

                    # 使用新cookies重新构建请求
                    if final_headers is None:
                        final_headers = {}
                    final_headers['Cookie'] = new_cookies

                    # 重新执行请求
                    logger.info("使用新cookies重新执行请求...")
                    response_data = self.process_http_request(
                        url=final_url, method=method, headers=final_headers, **kwargs)

                    if "error" in response_data:
                        return {"step": 1, "status": "error", "error": response_data["error"]}

                    # 再次检查状态码
                    if response_data.get("status_code") != 200:
                        error_msg = f"重新登录后请求仍然失败，状态码: {response_data.get('status_code')}"
                        logger.error(error_msg)
                        return {"step": 1, "status": "error", "error": error_msg}

                else:
                    logger.error("登录失败，无法获取新cookies")
                    return {"step": 1, "status": "error", "error": "登录失败，无法获取新cookies"}

            elif response_data.get("status_code") != 200:
                error_msg = f"HTTP请求失败，状态码: {response_data.get('status_code')}"
                logger.error(error_msg)
                return {"step": 1, "status": "error", "error": error_msg}

            # 2. 提取房型数据
            extracted_data = self.extract_room_types(response_data)

            if "error" in extracted_data:
                return {"step": 1, "status": "error", "error": extracted_data["error"]}

            # 3. 格式化为结构化数据
            formatted_data = self.format_room_types_for_prompt(extracted_data)

            # 4. 根据配置选择房型匹配方式（LLM直接调用 或 Agent方式）
            matching_result = self.process_room_matching(formatted_data)

            # 5. 解析大模型匹配结果并保存到CSV
            csv_path = ""
            if matching_result.get("status") == "success":
                # 获取大模型的回复
                llm_response = ""
                if matching_result.get("method") == "direct_llm_call":
                    llm_response = matching_result.get("llm_response", "")
                elif matching_result.get("method") == "agent_with_room_matching_tool":
                    llm_response = matching_result.get("agent_response", "")

                if llm_response:
                    logger.info("开始解析大模型匹配结果并保存到CSV...")

                    # 解析匹配结果
                    parsed_results = self.parse_llm_matching_result(
                        llm_response, hotel_id or "unknown")

                    # 保存到CSV文件
                    # parsed_results现在是一个字典，包含successful_matches和unsuccessful_matches
                    if parsed_results and (parsed_results.get('successful_matches') or parsed_results.get('unsuccessful_matches')):
                        csv_path = self.save_matching_results_to_csv(
                            parsed_results, hotel_id or "unknown")
                        logger.info(f"✅ 匹配结果已保存到CSV: {csv_path}")
                    else:
                        logger.warning("未能从大模型回复中解析出匹配结果")
                        # 即使没有解析到结果，也创建一个空的CSV文件记录
                        empty_results = {
                            'successful_matches': [], 'unsuccessful_matches': []}
                        csv_path = self.save_matching_results_to_csv(
                            empty_results, hotel_id or "unknown")

            # 计算解析结果统计
            matching_stats = {}
            if matching_result.get("status") == "success" and llm_response:
                parsed_results = self.parse_llm_matching_result(
                    llm_response, hotel_id or "unknown")

                successful_count = len(
                    parsed_results.get('successful_matches', []))
                unsuccessful_count = len(
                    parsed_results.get('unsuccessful_matches', []))
                total_processed = successful_count + unsuccessful_count

                matching_stats = {
                    "llm_successful_matches": successful_count,
                    "llm_unsuccessful_matches": unsuccessful_count,
                    "llm_total_processed": total_processed,
                    "llm_success_rate": f"{(successful_count / total_processed * 100):.1f}%" if total_processed > 0 else "0%"
                }

            # 构建统一的结果格式
            result = {
                "step": 1,
                "status": "success",
                "matching_method": matching_result.get("method", "unknown"),
                "matching_status": matching_result.get("status", "unknown"),
                "csv_output_path": csv_path,  # 新增CSV文件路径
                "room_data_summary": {
                    "hotel_info": extracted_data.get("hotel_info", {}),
                    "input_unmatched_count": len(extracted_data.get("unmatched_supplier_rooms", [])),
                    "input_standard_count": len(extracted_data.get("standard_room_types", [])),
                    "input_matched_pairs_count": len(extracted_data.get("matched_pairs", [])),
                    "formatted_data": formatted_data,
                    **matching_stats  # 展开LLM匹配统计
                },
                "metadata": {
                    "hotel_id": hotel_id,
                    "url": final_url,
                    "method": method,
                    "llm_model": self.llm_config.get('model_name'),
                    "configured_matching_method": self.room_matching_config.get('matching_method', 'agent')
                }
            }

            # 根据匹配方式添加相应的结果字段
            if matching_result.get("method") == "direct_llm_call":
                result.update({
                    "llm_response": matching_result.get("llm_response", ""),
                    "formatted_prompt": matching_result.get("formatted_prompt", ""),
                    "elapsed_time": matching_result.get("elapsed_time", 0)
                })
            elif matching_result.get("method") == "agent_with_room_matching_tool":
                result.update({
                    "agent_response": matching_result.get("agent_response", ""),
                    "agent_query": matching_result.get("agent_query", "")
                })

            # 如果匹配失败，添加错误信息
            if matching_result.get("status") == "error":
                result["matching_error"] = matching_result.get("error", "未知错误")

            logger.info("步骤1执行完成")
            return result

        except Exception as e:
            logger.error(f"步骤1执行失败: {str(e)}")
            return {"step": 1, "status": "error", "error": str(e)}

    def execute_batch_step(self, hotel_ids: List[str], method: str = "GET",
                           headers: Optional[Dict] = None, cookies: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """执行批量酒店的步骤1流程，并将结果合并到一个CSV文件

        Args:
            hotel_ids: 酒店ID列表
            method: HTTP方法
            headers: 自定义请求头
            cookies: 自定义cookies
            **kwargs: 其他参数
        """
        try:
            logger.info(f"开始批量处理步骤1，共{len(hotel_ids)}个酒店")

            all_hotel_results = []
            all_matching_results = []
            successful_count = 0
            failed_count = 0

            # 逐个处理每个酒店
            for i, hotel_id in enumerate(hotel_ids, 1):
                logger.info(f"处理酒店 {i}/{len(hotel_ids)}: {hotel_id}")

                try:
                    # 执行单个酒店的处理
                    single_result = self.execute_step(
                        hotel_id=hotel_id,
                        method=method,
                        headers=headers,
                        cookies=cookies,
                        **kwargs
                    )

                    # 添加酒店ID到结果中
                    single_result['hotel_id'] = hotel_id
                    all_hotel_results.append(single_result)

                    if single_result.get('status') == 'success':
                        successful_count += 1
                        logger.info(f"✅ 酒店 {hotel_id} 处理成功")

                        # 提取匹配结果用于合并
                        if single_result.get('csv_output_path'):
                            try:
                                import pandas as pd
                                df = pd.read_csv(
                                    single_result['csv_output_path'])
                                if len(df) > 0:
                                    all_matching_results.append(df)
                                    logger.info(
                                        f"从酒店 {hotel_id} 提取了 {len(df)} 条匹配结果")
                            except Exception as csv_error:
                                logger.warning(
                                    f"读取酒店 {hotel_id} 的CSV文件失败: {str(csv_error)}")
                    else:
                        failed_count += 1
                        error_msg = single_result.get('error', '未知错误')
                        logger.error(f"❌ 酒店 {hotel_id} 处理失败: {error_msg}")

                except Exception as hotel_error:
                    failed_count += 1
                    error_msg = str(hotel_error)
                    logger.error(f"❌ 酒店 {hotel_id} 处理异常: {error_msg}")
                    all_hotel_results.append({
                        'hotel_id': hotel_id,
                        'status': 'error',
                        'error': error_msg,
                        'step': 1
                    })

            # 合并所有匹配结果到一个CSV文件
            merged_csv_path = ""
            if all_matching_results:
                try:
                    # 合并所有DataFrame
                    merged_df = pd.concat(
                        all_matching_results, ignore_index=True)

                    # 生成合并CSV文件路径
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = os.path.join(
                        os.path.dirname(__file__), '..', 'output')
                    os.makedirs(output_dir, exist_ok=True)

                    csv_filename = f"batch_matching_results_{len(hotel_ids)}hotels_{timestamp}.csv"
                    merged_csv_path = os.path.join(output_dir, csv_filename)

                    # 保存合并的CSV文件
                    merged_df.to_csv(
                        merged_csv_path, index=False, encoding='utf-8-sig')

                    logger.info(f"✅ 所有酒店匹配结果已合并保存到: {merged_csv_path}")
                    logger.info(f"合并CSV包含 {len(merged_df)} 条记录")

                    # 统计各酒店的记录数
                    if '酒店ID' in merged_df.columns:
                        hotel_counts = merged_df['酒店ID'].value_counts()
                        logger.info("各酒店记录数统计:")
                        for hotel_id, count in hotel_counts.items():
                            logger.info(f"  酒店 {hotel_id}: {count} 条")

                except Exception as merge_error:
                    logger.error(f"合并CSV文件失败: {str(merge_error)}")
                    merged_csv_path = ""
            else:
                logger.warning("没有找到任何匹配结果，无法创建合并CSV文件")

            # 计算成功率
            total_hotels = len(hotel_ids)
            success_rate = f"{(successful_count / total_hotels * 100):.1f}%" if total_hotels > 0 else "0%"

            # 构建批量处理结果
            batch_result = {
                "step": 1,
                "status": "success",
                "batch_summary": {
                    "total_hotels": total_hotels,
                    "successful_hotels": successful_count,
                    "failed_hotels": failed_count,
                    "success_rate": success_rate,
                    "processed_hotel_ids": hotel_ids
                },
                "hotel_results": all_hotel_results,
                "merged_csv_path": merged_csv_path,
                "metadata": {
                    "batch_processing": True,
                    "method": method,
                    "llm_model": self.llm_config.get('model_name'),
                    "configured_matching_method": self.room_matching_config.get('matching_method', 'agent')
                }
            }

            logger.info(f"✅ 批量处理完成: 成功 {successful_count}/{total_hotels} 个酒店")
            return batch_result

        except Exception as e:
            logger.error(f"批量步骤1执行失败: {str(e)}")
            return {
                "step": 1,
                "status": "error",
                "error": str(e),
                "batch_summary": {
                    "total_hotels": len(hotel_ids) if hotel_ids else 0,
                    "successful_hotels": 0,
                    "failed_hotels": len(hotel_ids) if hotel_ids else 0,
                    "success_rate": "0%"
                }
            }
