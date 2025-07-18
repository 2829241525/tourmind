#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习房型匹配工具
调用外部API进行房型匹配相似度计算
"""

import requests
import json
import logging
import csv
import os
import hashlib
import glob
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

# 配置信息
DL_MATCH_CONFIG = {
    "api_url": "http://10.0.0.110:13001/match/",
    "headers": {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Connection': 'keep-alive'
    },
    "token": "tourmind_room_match",
    "timeout": 30,
    "csv_filename_prefix": "room_similarity_"
}

logger = logging.getLogger(__name__)


   
class RoomMatchingTool(BaseTool):
    """房型匹配工具 - 供Agent调用"""
    name: str = "room_matching"
    description: str = """
    用于房型匹配相似度计算。
    如果某条供应商房型，疑似匹配的标准房型超过1个,只将超过一个的房型传入，用于辅助判断。
    输入格式如下：
    1.供应商房型：房型名称，疑似匹配标准房型[1]xxx,[2]xxx,[3]xxx,[4]xxx,[5]xxx,
    2.供应商房型：房型名称，疑似匹配标准房型[1]xxx,[2]xxx,[3]xxx,[4]xxx,[5]xxx,
    
    例如：
    1.供应商房型：King Room Deluxe，疑似匹配标准房型[1]King Room Standard,[2]Double Room,[3]Suite Room,[4]King Room,[5]Queen Room
    2.供应商房型：Queen Suite，疑似匹配标准房型[1]Queen Room,[2]Suite Room
    """
    
    def __init__(self, matcher_config: Optional[Dict] = None):
        """初始化工具
        
        Args:
            matcher_config: DL匹配器配置
        """
        super().__init__()
        self._matcher_config = matcher_config or DL_MATCH_CONFIG
        
    def _parse_input(self, tool_input: Union[str, dict], tool_call_id: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """解析输入并提取房型数据
        
        Args:
            tool_input: Agent传入的输入数据
            tool_call_id: 工具调用ID（BaseTool要求的参数）
            
        Returns:
            解析后的数据，返回字符串让_run接收位置参数
        """
        try:
            # 如果输入是字符串，直接返回（向后兼容）
            if isinstance(tool_input, str):
                return tool_input
            
            # 如果输入是字典，也直接返回让_run处理
            if isinstance(tool_input, dict):
                return tool_input
            
            # 其他类型，返回错误字符串
            return f"不支持的输入类型: {type(tool_input)}"
            
        except Exception as e:
            logger.error(f"输入解析失败: {str(e)}")
            return f"输入解析失败: {str(e)}"
    
    def _parse_string_input(self, query: str) -> Dict[str, Any]:
        """解析字符串格式的输入
        
        Args:
            query: 输入的查询字符串
            
        Returns:
            解析后的房型数据，保持供应商房型与其疑似匹配房型的对应关系
        """
        lines = query.strip().split('\n')
        room_pairs = []  # 存储[(供应商房型, [对应的标准房型列表]), ...]
        
        for line in lines:
            line = line.strip()
            
            # 查找"供应商房型："
            if "供应商房型：" not in line:
                logger.warning(f"行内容'{line}'格式不正确，跳过")
                continue
            
            # 分割供应商房型和标准房型部分
            parts = line.split("疑似匹配标准房型")
            if len(parts) != 2:
                logger.warning(f"行内容'{line}'格式不正确，跳过")
                continue
            
            # 提取供应商房型
            supplier_part = parts[0]
            supplier_room = supplier_part.split("供应商房型：")[1].replace("，", "").strip()
            
            # 提取对应的标准房型列表
            standard_part = parts[1]
            # 使用正则表达式提取[数字]房型名称的模式
            pattern = r'\[(\d+)\]([^，\[\]]+)'
            matches = re.findall(pattern, standard_part)
            
            standard_rooms_for_this_supplier = []
            for _, room_name in matches:
                standard_rooms_for_this_supplier.append(room_name.strip())
            
            if standard_rooms_for_this_supplier:
                room_pairs.append((supplier_room, standard_rooms_for_this_supplier))
        
        logger.info(f"从字符串解析：{len(room_pairs)}个供应商房型及其对应的疑似匹配房型")
        
        return {
            "room_pairs": room_pairs,
            "total_supplier": len(room_pairs)
        }
    
    def _run(
        self, 
        tool_input: Union[str, dict] = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """执行房型匹配
        
        Args:
            tool_input: 输入数据
            run_manager: 回调管理器
            
        Returns:
            匹配结果字符串
        """
        try:
            # 如果是错误字符串，直接返回
            if isinstance(tool_input, str) and tool_input.startswith(("不支持的输入类型", "输入解析失败")):
                return tool_input
            
            # 解析输入数据
            if isinstance(tool_input, str):
                # 字符串输入，使用字符串解析方法
                parsed_data = self._parse_string_input(tool_input)
            elif isinstance(tool_input, dict):
                # 字典输入，需要转换为room_pairs格式
                if 'supplier_rooms' in tool_input and 'standard_rooms' in tool_input:
                    supplier_rooms = tool_input['supplier_rooms']
                    standard_rooms = tool_input['standard_rooms']
                    # 转换为room_pairs格式（每个供应商房型对应所有标准房型）
                    room_pairs = [(room, standard_rooms) for room in supplier_rooms]
                    parsed_data = {"room_pairs": room_pairs, "total_supplier": len(room_pairs)}
                else:
                    return "错误：字典输入中缺少必要的房型数据字段"
            else:
                return f"错误：不支持的输入类型 {type(tool_input)}"
            
            # 检查解析结果
            if "error" in parsed_data:
                return f"输入解析错误：{parsed_data['error']}"
            
            room_pairs = parsed_data.get("room_pairs", [])
            
            # 验证房型数据
            if not room_pairs:
                return "错误：未能解析出有效的房型数据。请检查输入格式。"
            
            logger.info(f"开始房型匹配：{len(room_pairs)}个供应商房型，每个都有对应的疑似匹配房型")
            
            # 创建匹配器并计算相似度
            matcher = create_dl_matcher(self._matcher_config)
            result = matcher.calculate_paired_similarities(room_pairs)
            
            if result["status"] != "success":
                return f"房型匹配失败: {result.get('error', 'Unknown error')}"
            
            # 格式化输出结果
            return self._format_paired_match_result(result, matcher.similarity_results)
            
        except Exception as e:
            error_msg = f"房型匹配工具执行失败: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _format_paired_match_result(self, result: Dict[str, Any], similarity_results: List[Dict]) -> str:
        """格式化配对匹配结果，智能判断相似度差异
        
        Args:
            result: API返回的完整结果
            similarity_results: 相似度结果列表
            
        Returns:
            格式化的结果字符串，包含智能判断逻辑
        """
        if not similarity_results:
            return "未找到匹配的标准房型"
        
        # 按供应商房型分组，获取每个供应商房型的所有匹配结果
        supplier_matches = {}
        for result_item in similarity_results:
            supplier_room = result_item["supplier_room"]
            standard_room = result_item["standard_room"]
            similarity = result_item["similarity"]
            
            if supplier_room not in supplier_matches:
                supplier_matches[supplier_room] = []
            
            # 验证standard_room是否已经存在，避免重复添加
            existing_standard_rooms = [match["standard_room"] for match in supplier_matches[supplier_room]]
            if standard_room not in existing_standard_rooms:
                supplier_matches[supplier_room].append({
                    "standard_room": standard_room,
                    "similarity": similarity
                })
            else:
                logger.debug(f"跳过重复的标准房型: {supplier_room} → {standard_room}")
        
        # 对每个供应商房型的匹配结果按相似度排序
        for supplier_room in supplier_matches:
            supplier_matches[supplier_room].sort(key=lambda x: x["similarity"], reverse=True)
        
        output_lines = []
        output_lines.append("房型最佳匹配结果：")
        
        for i, (supplier_room, matches) in enumerate(supplier_matches.items(), 1):
            output_lines.append(f"\n{i}. 供应商房型：{supplier_room}")
            
            if len(matches) == 1:
                # 只有一个匹配结果，直接返回
                output_lines.append(f"疑似匹配的标准房型中，相似度最高的为：{matches[0]['standard_room']}")
            elif len(matches) >= 2:
                # 有多个匹配结果，比较最高和次高相似度
                highest_similarity = matches[0]["similarity"]
                second_highest_similarity = matches[1]["similarity"]
                similarity_diff = highest_similarity - second_highest_similarity
                
                if similarity_diff > 0.01:
                    # 相似度差异明显，返回最高的
                    output_lines.append(f"疑似匹配的标准房型中，相似度最高的为：{matches[0]['standard_room']}")
                else:
                    # 相似度差异不明显，提示用户自行判断
                    output_lines.append("疑似匹配的标准房型中，没有相似度明显更高者，请根据规则判断哪一条更合适")
                    output_lines.append("前几个相似度接近的选项：")
                    
                    # 显示相似度差异小于等于0.01的所有选项
                    similar_options = []
                    for match in matches:
                        if highest_similarity - match["similarity"] <= 0.01:
                            similar_options.append(match["standard_room"])
                        else:
                            break  # 由于已排序，后面的相似度差异会更大
                    
                    for j, option in enumerate(similar_options, 1):
                        output_lines.append(f"   选项{j}: {option}")

        output_lines.append("该相似度计算可能存在错误，仅作参考，请根据你的推理进行判断，我不再提供更多的信息，请按照格式要求，给出最终的匹配结果")
        return "\n".join(output_lines)
    
    async def _arun(
        self, 
        tool_input: Union[str, dict] = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """异步执行房型匹配"""
        return self._run(
            tool_input=tool_input,
            run_manager=run_manager,
            **kwargs
        )

def create_room_matching_tool(config: Optional[Dict] = None) -> RoomMatchingTool:
        """创建房型匹配工具实例
        
        Args:
            config: 自定义配置
            
        Returns:
            RoomMatchingTool实例
        """
        return RoomMatchingTool(matcher_config=config)


class DLRoomMatcher:
    """深度学习房型匹配器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化匹配器
        
        Args:
            config: 自定义配置，如果为None则使用默认配置
        """
        self.config = config or DL_MATCH_CONFIG
        self.similarity_results = []
        
        # 确保输出目录存在
        os.makedirs(self.config.get("csv_output_dir", "logs"), exist_ok=True)
        
    def call_match_api(self, spl_room_names: List[str], s_room_names: List[str], 
                      spl_room_bed_names: Optional[List[str]] = None, 
                      s_room_bed_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """调用房型匹配API
        
        Args:
            spl_room_names: 供应商房型名称列表
            s_room_names: 标准房型名称列表
            spl_room_bed_names: 供应商床型名称列表（可选）
            s_room_bed_names: 标准床型名称列表（可选）
            
        Returns:
            API响应结果
        """
        try:
            # 验证数组长度是否相同
            if len(spl_room_names) != len(s_room_names):
                raise ValueError(f"供应商房型数组长度({len(spl_room_names)})与标准房型数组长度({len(s_room_names)})不匹配")
            
            array_length = len(spl_room_names)
            
            # 构建请求数据，确保所有数组长度相同
            request_data = {
                "spl_room_names": spl_room_names,
                "spl_room_bed_names": spl_room_bed_names or [""] * array_length,
                "s_room_names": s_room_names,
                "s_room_bed_names": s_room_bed_names or [""] * array_length,
                "token": self.config["token"]
            }
            
            # 验证所有数组长度是否一致
            all_lengths = [
                len(request_data["spl_room_names"]),
                len(request_data["spl_room_bed_names"]),
                len(request_data["s_room_names"]),
                len(request_data["s_room_bed_names"])
            ]
            
            if len(set(all_lengths)) != 1:
                raise ValueError(f"所有数组长度必须相同，当前长度: {all_lengths}")
            
            logger.info(f"调用房型匹配API，总配对数量: {array_length}（所有数组长度相同）")
            
            # 发送POST请求
            response = requests.post(
                url=self.config["api_url"],
                headers=self.config["headers"],
                json=request_data,
                timeout=self.config.get("timeout", 30)
            )
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json()
                logger.info(f"API调用成功，返回相似度数量: {len(result.get('similarities', []))}")
                return {
                    "status": "success",
                    "data": result,
                    "request_data": request_data
                }
            else:
                error_msg = f"API调用失败，状态码: {response.status_code}, 响应: {response.text}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "request_data": request_data
                }
                
        except Exception as e:
            error_msg = f"API调用异常: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "request_data": request_data if 'request_data' in locals() else None
            }
    
    def calculate_paired_similarities(self, room_pairs: List[tuple]) -> Dict[str, Any]:
        """计算成对的房型相似度（每个供应商房型只与其对应的疑似匹配房型计算）
        
        Args:
            room_pairs: [(供应商房型, [对应的标准房型列表]), ...]
            
        Returns:
            包含相似度结果和统计信息的结果
        """
        try:
            if not room_pairs:
                logger.warning("房型配对列表为空，无法计算相似度")
                return {"status": "error", "error": "房型配对列表为空"}
            
            # 构建配对列表，每个供应商房型只与其对应的疑似匹配房型计算
            expanded_spl_rooms = []
            expanded_std_rooms = []
            
            for supplier_room, standard_room_list in room_pairs:
                for std_room in standard_room_list:
                    expanded_spl_rooms.append(supplier_room)
                    expanded_std_rooms.append(std_room)
            
            total_pairs = len(expanded_spl_rooms)
            logger.info(f"构建配对列表：{len(room_pairs)}个供应商房型对应其疑似匹配房型，共{total_pairs}个配对")
            
            # 调用API计算相似度
            api_result = self.call_match_api(
                spl_room_names=expanded_spl_rooms,
                s_room_names=expanded_std_rooms
            )
            
            if api_result["status"] != "success":
                return api_result
            
            similarities = api_result["data"].get("similarities", [])
            
            if not similarities:
                return {"status": "error", "error": "API返回的相似度为空"}
            
            if len(similarities) != total_pairs:
                logger.warning(f"API返回的相似度数量({len(similarities)})与期望的配对数量({total_pairs})不匹配")
            
            # 构建结果，按供应商房型分组
            similarity_results = []
            index = 0
            
            for supplier_room, standard_room_list in room_pairs:
                for std_room in standard_room_list:
                    if index < len(similarities):
                        similarity = similarities[index]
                        similarity_results.append({
                            "supplier_room": supplier_room,
                            "standard_room": std_room,
                            "similarity": similarity
                        })
                        index += 1
                    else:
                        logger.warning(f"相似度数据不足，索引{index}超出范围")
                        break
            
            # 保存结果到实例变量
            self.similarity_results = similarity_results
            
            # 计算统计信息
            all_similarities = [item["similarity"] for item in similarity_results]
            stats = {
                "total_pairs": len(all_similarities),
                "avg_similarity": sum(all_similarities) / len(all_similarities) if all_similarities else 0,
                "max_similarity": max(all_similarities) if all_similarities else 0,
                "min_similarity": min(all_similarities) if all_similarities else 0
            }
            
            logger.info(f"配对相似度计算完成，共{stats['total_pairs']}个配对，平均相似度: {stats['avg_similarity']:.4f}")
            
            csv_path = self.save_to_csv()
            print(f"结果已保存到: {csv_path}")
            return {
                "status": "success",
                "similarity_results": similarity_results,
                "statistics": stats,
                "original_similarities": similarities,
                "room_pairs_info": {
                    "original_supplier_count": len(room_pairs),
                    "total_pairs": total_pairs
                }
            }
            
        except Exception as e:
            error_msg = f"配对相似度计算失败: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

    def calculate_similarities(self, unmatched_rooms: List[str], standard_rooms: List[str]) -> Dict[str, Any]:
        """计算未匹配房型与标准房型之间的相似度
        
        Args:
            unmatched_rooms: 未匹配的供应商房型列表
            standard_rooms: 标准房型列表
            
        Returns:
            包含相似度矩阵和统计信息的结果
        """
        try:
            if not unmatched_rooms or not standard_rooms:
                logger.warning("房型列表为空，无法计算相似度")
                return {"status": "error", "error": "房型列表为空"}
            
            # 构建笛卡尔积，确保所有数组长度相同
            expanded_spl_rooms = []
            expanded_std_rooms = []
            
            for spl_room in unmatched_rooms:
                for std_room in standard_rooms:
                    expanded_spl_rooms.append(spl_room)
                    expanded_std_rooms.append(std_room)
            
            total_pairs = len(expanded_spl_rooms)
            logger.info(f"构建笛卡尔积：{len(unmatched_rooms)}个供应商房型 × {len(standard_rooms)}个标准房型 = {total_pairs}个配对")
            
            # 调用API计算相似度
            api_result = self.call_match_api(
                spl_room_names=expanded_spl_rooms,
                s_room_names=expanded_std_rooms
            )
            
            if api_result["status"] != "success":
                return api_result
            
            similarities = api_result["data"].get("similarities", [])
            
            if not similarities:
                return {"status": "error", "error": "API返回的相似度为空"}
            
            if len(similarities) != total_pairs:
                logger.warning(f"API返回的相似度数量({len(similarities)})与期望的配对数量({total_pairs})不匹配")
            
            # 构建相似度矩阵结果
            similarity_matrix = []
            index = 0
            
            for i, spl_room in enumerate(unmatched_rooms):
                room_similarities = []
                for j, std_room in enumerate(standard_rooms):
                    if index < len(similarities):
                        similarity = similarities[index]
                        room_similarities.append({
                            "supplier_room": spl_room,
                            "standard_room": std_room,
                            "similarity": similarity,
                            "supplier_index": i,
                            "standard_index": j
                        })
                        index += 1
                    else:
                        logger.warning(f"相似度数据不足，索引{index}超出范围")
                        break
                similarity_matrix.append(room_similarities)
            
            # 保存结果到实例变量
            self.similarity_results = [item for sublist in similarity_matrix for item in sublist]
            
            # 计算统计信息
            all_similarities = [item["similarity"] for item in self.similarity_results]
            stats = {
                "total_pairs": len(all_similarities),
                "avg_similarity": sum(all_similarities) / len(all_similarities) if all_similarities else 0,
                "max_similarity": max(all_similarities) if all_similarities else 0,
                "min_similarity": min(all_similarities) if all_similarities else 0
            }
            
            logger.info(f"相似度计算完成，共{stats['total_pairs']}个配对，平均相似度: {stats['avg_similarity']:.4f}")
            
            csv_path = self.save_to_csv()
            print(f"结果已保存到: {csv_path}")
            return {
                "status": "success",
                "similarity_matrix": similarity_matrix,
                "similarity_list": self.similarity_results,
                "statistics": stats,
                "original_similarities": similarities,
                "cartesian_product_info": {
                    "original_supplier_count": len(unmatched_rooms),
                    "original_standard_count": len(standard_rooms),
                    "total_pairs": total_pairs
                }
            }
            
        except Exception as e:
            error_msg = f"相似度计算失败: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    def _calculate_content_hash(self) -> str:
        """计算相似度结果内容的hash值
        
        Returns:
            内容的MD5 hash值
        """
        # 创建用于hash计算的数据字符串
        content_data = []
        for result in sorted(self.similarity_results, key=lambda x: (x['supplier_room'], x['standard_room'])):
            content_data.append(f"{result['supplier_room']}|{result['standard_room']}|{result['similarity']}")
        
        content_string = '\n'.join(content_data)
        return hashlib.md5(content_string.encode('utf-8')).hexdigest()[:8]
    
    def _find_existing_file(self, content_hash: str) -> Optional[str]:
        """查找已存在的相同内容文件
        
        Args:
            content_hash: 内容hash值
            
        Returns:
            已存在文件的路径，如果不存在返回None
        """
        output_dir = self.config.get("csv_output_dir", "logs")
        pattern = os.path.join(output_dir, f"*{content_hash}*.csv")
        existing_files = glob.glob(pattern)
        
        if existing_files:
            # 返回最新的文件
            return max(existing_files, key=os.path.getmtime)
        return None
    
    def save_to_csv(self, filename: Optional[str] = None) -> str:
        """将相似度结果保存到CSV文件，支持内容去重
        
        Args:
            filename: 自定义文件名，如果为None则自动生成
            
        Returns:
            保存的文件路径
        """
        try:
            if not self.similarity_results:
                logger.warning("没有相似度结果可以保存")
                return ""
            
            # 计算内容hash值
            content_hash = self._calculate_content_hash()
            
            # 检查是否已存在相同内容的文件
            existing_file = self._find_existing_file(content_hash)
            if existing_file and os.path.exists(existing_file):
                logger.info(f"发现相同内容的文件已存在: {existing_file}，跳过保存")
                return existing_file
            
            # 生成文件名（包含hash值）
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.config.get('csv_filename_prefix', 'room_similarity_')}{content_hash}_{timestamp}.csv"
            
            # 完整文件路径
            output_dir = self.config.get("csv_output_dir", "logs")
            filepath = os.path.join(output_dir, filename)
            
            # 写入CSV文件
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['supplier_room', 'standard_room', 'similarity', 'supplier_index', 'standard_index']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 写入头部
                writer.writeheader()
                
                # 写入数据
                for result in self.similarity_results:
                    writer.writerow(result)
            
            logger.info(f"相似度结果已保存到: {filepath}（内容hash: {content_hash}）")
            return filepath
            
        except Exception as e:
            error_msg = f"保存CSV文件失败: {str(e)}"
            logger.error(error_msg)
            return ""
    
    def get_best_matches(self, threshold: float = 0.8) -> List[Dict]:
        """获取相似度高于阈值的最佳匹配
        
        Args:
            threshold: 相似度阈值
            
        Returns:
            最佳匹配列表
        """
        if not self.similarity_results:
            return []
        
        # 按供应商房型分组，获取每个供应商房型的最佳匹配
        best_matches = {}
        for result in self.similarity_results:
            supplier_room = result["supplier_room"]
            if supplier_room not in best_matches or result["similarity"] > best_matches[supplier_room]["similarity"]:
                best_matches[supplier_room] = result
        
        # 过滤高于阈值的匹配
        high_quality_matches = [
            match for match in best_matches.values() 
            if match["similarity"] >= threshold
        ]
        
        # 按相似度排序
        high_quality_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        logger.info(f"找到{len(high_quality_matches)}个相似度高于{threshold}的最佳匹配")
        return high_quality_matches


def create_dl_matcher(config: Optional[Dict] = None) -> DLRoomMatcher:
    """创建DL房型匹配器实例
    
    Args:
        config: 自定义配置
        
    Returns:
        DLRoomMatcher实例
    """
    return DLRoomMatcher(config)


# 测试函数
def test_dl_matcher():
    """测试DL匹配器功能"""
    # 测试数据 - 使用新的配对格式
    test_room_pairs = [
        ("King Room - Disability Access/Non-Smoking", 
         ["1 King Hearing Accessible Non-Smoking 1 king bed", "1 King Bed Non-Smoking 1 king bed"]),
        ("Double Room Non-Smoking - 2 Double Beds", 
         ["2 Double Beds Non-Smoking", "Double Room Standard"])
    ]
    
    print(f"测试数据：{len(test_room_pairs)}个供应商房型及其对应的疑似匹配房型")
    total_pairs = sum(len(std_rooms) for _, std_rooms in test_room_pairs)
    print(f"总计算配对数：{total_pairs}个（非笛卡尔积）")
    
    # 创建匹配器
    matcher = create_dl_matcher()
    
    # 计算配对相似度
    result = matcher.calculate_paired_similarities(test_room_pairs)
    
    if result["status"] == "success":
        print("API调用成功！")
        room_pairs_info = result.get('room_pairs_info', {})
        print(f"配对信息: {room_pairs_info}")
        print(f"相似度数量: {len(result['original_similarities'])}")
        print(f"前几个相似度值: {result['original_similarities'][:4] if len(result['original_similarities']) >= 4 else result['original_similarities']}")
        
        # 显示相似度结果结构
        if result['similarity_results']:
            print(f"相似度配对示例:")
            for i, item in enumerate(result['similarity_results'][:4]):  # 显示前4个配对
                print(f"  [{i+1}] {item['supplier_room'][:50]}... → {item['standard_room'][:50]}... (相似度: {item['similarity']:.4f})")
        
        # 测试工具输出格式
        tool = create_room_matching_tool()
        formatted_result = tool._format_paired_match_result(result, matcher.similarity_results)
        print(f"\n工具输出格式:")
        print(formatted_result)
        
        return True
    else:
        print(f"API调用失败: {result.get('error', 'Unknown error')}")
        return False


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    test_dl_matcher()



 
