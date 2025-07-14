#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房型匹配项目 - 房源匹配引擎
执行房源与用户需求的智能匹配
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

from langchain.chains import LLMChain
from ..prompts import get_prompt
from ..config import setup_logging

class RoomMatchingEngine:
    """房源匹配引擎"""
    
    def __init__(self, llm, config: Dict[str, Any] = None):
        """
        初始化匹配引擎
        
        Args:
            llm: 语言模型实例
            config: 配置参数
        """
        self.llm = llm
        self.config = config or {}
        self.logger = setup_logging()
        self.prompt = get_prompt('matching_evaluation')
        
        # 匹配链
        self.matching_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=self.config.get('verbose', False)
        )
        
        # 匹配策略配置
        self.matching_strategies = {
            'hybrid': self._hybrid_matching,
            'semantic': self._semantic_matching,
            'rule_based': self._rule_based_matching,
            'ml_enhanced': self._ml_enhanced_matching
        }
        
        # 匹配权重
        self.matching_weights = {
            'room_type_match': 0.25,
            'location_match': 0.30,
            'price_match': 0.25,
            'facility_match': 0.20
        }
    
    def match_rooms(self, 
                   room_list: List[Dict[str, Any]],
                   user_requirements: Dict[str, Any],
                   matching_strategy: str = 'hybrid',
                   max_results: int = 10,
                   filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行房源匹配
        
        Args:
            room_list: 房源列表
            user_requirements: 用户需求分析结果
            matching_strategy: 匹配策略
            max_results: 最大返回结果数
            filters: 额外过滤条件
            
        Returns:
            Dict: 匹配结果
        """
        try:
            self.logger.info(f"开始执行房源匹配，策略: {matching_strategy}")
            
            # 预处理房源数据
            processed_rooms = self._preprocess_rooms(room_list, filters)
            
            # 获取匹配策略
            matching_func = self.matching_strategies.get(
                matching_strategy, self._hybrid_matching
            )
            
            # 执行匹配
            matching_results = matching_func(processed_rooms, user_requirements)
            
            # 排序和筛选
            sorted_results = self._sort_and_filter_results(
                matching_results, max_results
            )
            
            # 计算匹配统计
            matching_stats = self._calculate_matching_stats(
                sorted_results, user_requirements
            )
            
            # 构建最终结果
            final_result = {
                'matched_rooms': sorted_results,
                'matching_stats': matching_stats,
                'matching_metadata': {
                    'strategy': matching_strategy,
                    'total_candidates': len(processed_rooms),
                    'returned_results': len(sorted_results),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"房源匹配完成，返回{len(sorted_results)}个结果")
            return final_result
            
        except Exception as e:
            self.logger.error(f"房源匹配失败: {str(e)}")
            return self._get_error_result(str(e))
    
    def evaluate_single_match(self, 
                             room_info: Dict[str, Any],
                             user_requirements: Dict[str, Any],
                             detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        评估单个房源的匹配度
        
        Args:
            room_info: 房源信息
            user_requirements: 用户需求
            detailed_analysis: 是否进行详细分析
            
        Returns:
            Dict: 匹配评估结果
        """
        try:
            self.logger.info("开始单个房源匹配评估")
            
            # 基础匹配度计算
            basic_scores = self._calculate_basic_matching_scores(
                room_info, user_requirements
            )
            
            # LLM增强评估
            if detailed_analysis:
                llm_evaluation = self._llm_enhanced_evaluation(
                    room_info, user_requirements
                )
                basic_scores.update(llm_evaluation)
            
            # 综合匹配度计算
            overall_score = self._calculate_overall_score(basic_scores)
            
            # 匹配解释生成
            match_explanation = self._generate_match_explanation(
                basic_scores, room_info, user_requirements
            )
            
            result = {
                'room_id': room_info.get('id', 'unknown'),
                'overall_score': overall_score,
                'detailed_scores': basic_scores,
                'match_explanation': match_explanation,
                'recommendation_level': self._get_recommendation_level(overall_score),
                'evaluation_metadata': {
                    'detailed_analysis': detailed_analysis,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            self.logger.info("单个房源匹配评估完成")
            return result
            
        except Exception as e:
            self.logger.error(f"单个房源匹配评估失败: {str(e)}")
            return self._get_error_result(str(e))
    
    def _preprocess_rooms(self, 
                         room_list: List[Dict[str, Any]], 
                         filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """预处理房源数据"""
        processed_rooms = []
        
        for room in room_list:
            # 数据清洗
            cleaned_room = self._clean_room_data(room)
            
            # 应用过滤条件
            if filters and not self._apply_filters(cleaned_room, filters):
                continue
                
            # 标准化数据格式
            standardized_room = self._standardize_room_format(cleaned_room)
            processed_rooms.append(standardized_room)
        
        return processed_rooms
    
    def _hybrid_matching(self, 
                        rooms: List[Dict[str, Any]], 
                        requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """混合匹配策略"""
        results = []
        
        for room in rooms:
            # 规则匹配
            rule_score = self._rule_based_score(room, requirements)
            
            # 语义匹配
            semantic_score = self._semantic_similarity_score(room, requirements)
            
            # 综合评分
            hybrid_score = 0.6 * rule_score + 0.4 * semantic_score
            
            results.append({
                'room_data': room,
                'matching_score': hybrid_score,
                'rule_score': rule_score,
                'semantic_score': semantic_score,
                'matching_method': 'hybrid'
            })
        
        return results
    
    def _semantic_matching(self, 
                          rooms: List[Dict[str, Any]], 
                          requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """语义匹配策略"""
        results = []
        
        for room in rooms:
            semantic_score = self._semantic_similarity_score(room, requirements)
            
            results.append({
                'room_data': room,
                'matching_score': semantic_score,
                'matching_method': 'semantic'
            })
        
        return results
    
    def _rule_based_matching(self, 
                           rooms: List[Dict[str, Any]], 
                           requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于规则的匹配策略"""
        results = []
        
        for room in rooms:
            rule_score = self._rule_based_score(room, requirements)
            
            results.append({
                'room_data': room,
                'matching_score': rule_score,
                'matching_method': 'rule_based'
            })
        
        return results
    
    def _ml_enhanced_matching(self, 
                            rooms: List[Dict[str, Any]], 
                            requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """机器学习增强匹配策略"""
        # 预留ML模型接口
        results = []
        
        for room in rooms:
            # 特征向量化
            room_features = self._extract_room_features(room)
            requirement_features = self._extract_requirement_features(requirements)
            
            # ML模型预测（这里用简化逻辑代替）
            ml_score = self._simple_ml_score(room_features, requirement_features)
            
            results.append({
                'room_data': room,
                'matching_score': ml_score,
                'matching_method': 'ml_enhanced'
            })
        
        return results
    
    def _calculate_basic_matching_scores(self, 
                                       room_info: Dict[str, Any], 
                                       requirements: Dict[str, Any]) -> Dict[str, float]:
        """计算基础匹配分数"""
        scores = {}
        
        # 房型匹配度
        scores['room_type_match'] = self._calculate_room_type_match(room_info, requirements)
        
        # 位置匹配度
        scores['location_match'] = self._calculate_location_match(room_info, requirements)
        
        # 价格匹配度
        scores['price_match'] = self._calculate_price_match(room_info, requirements)
        
        # 设施匹配度
        scores['facility_match'] = self._calculate_facility_match(room_info, requirements)
        
        return scores
    
    def _llm_enhanced_evaluation(self, 
                               room_info: Dict[str, Any], 
                               requirements: Dict[str, Any]) -> Dict[str, Any]:
        """LLM增强评估"""
        try:
            # 构建匹配标准
            matching_criteria = self._build_matching_criteria(requirements)
            
            # 执行LLM评估
            evaluation_result = self.matching_chain.run(
                room_info=json.dumps(room_info, ensure_ascii=False),
                user_requirements=json.dumps(requirements.get('parsed_requirements', {}), ensure_ascii=False),
                matching_criteria=matching_criteria
            )
            
            # 解析评估结果
            parsed_evaluation = self._parse_llm_evaluation(evaluation_result)
            
            return {
                'llm_evaluation': parsed_evaluation,
                'llm_overall_score': parsed_evaluation.get('overall_score', 0.5),
                'llm_reasoning': parsed_evaluation.get('reasoning', '')
            }
            
        except Exception as e:
            self.logger.warning(f"LLM增强评估失败: {str(e)}")
            return {
                'llm_evaluation': None,
                'llm_overall_score': 0.5,
                'llm_reasoning': 'LLM评估失败'
            }
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """计算综合匹配分数"""
        weighted_sum = 0
        total_weight = 0
        
        for score_type, weight in self.matching_weights.items():
            if score_type in scores:
                weighted_sum += scores[score_type] * weight
                total_weight += weight
        
        # 如果有LLM评估，加入权重
        if 'llm_overall_score' in scores:
            weighted_sum += scores['llm_overall_score'] * 0.3
            total_weight += 0.3
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _sort_and_filter_results(self, 
                               results: List[Dict[str, Any]], 
                               max_results: int) -> List[Dict[str, Any]]:
        """排序和过滤结果"""
        # 按匹配分数排序
        sorted_results = sorted(
            results, 
            key=lambda x: x.get('matching_score', 0), 
            reverse=True
        )
        
        # 限制结果数量
        return sorted_results[:max_results]
    
    def _calculate_matching_stats(self, 
                                results: List[Dict[str, Any]], 
                                requirements: Dict[str, Any]) -> Dict[str, Any]:
        """计算匹配统计信息"""
        if not results:
            return {'average_score': 0, 'score_distribution': {}}
        
        scores = [r.get('matching_score', 0) for r in results]
        
        return {
            'average_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'score_distribution': self._calculate_score_distribution(scores),
            'high_quality_matches': len([s for s in scores if s >= 0.8])
        }
    
    # ===== 辅助方法 =====
    
    def _clean_room_data(self, room: Dict[str, Any]) -> Dict[str, Any]:
        """清洗房源数据"""
        # 实现数据清洗逻辑
        return room
    
    def _apply_filters(self, room: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """应用过滤条件"""
        # 实现过滤逻辑
        return True
    
    def _standardize_room_format(self, room: Dict[str, Any]) -> Dict[str, Any]:
        """标准化房源格式"""
        # 实现格式标准化逻辑
        return room
    
    def _rule_based_score(self, room: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """基于规则的评分"""
        # 实现规则评分逻辑
        return 0.7
    
    def _semantic_similarity_score(self, room: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """语义相似度评分"""
        # 实现语义相似度计算
        return 0.6
    
    def _extract_room_features(self, room: Dict[str, Any]) -> np.ndarray:
        """提取房源特征向量"""
        # 实现特征提取逻辑
        return np.array([1.0, 2.0, 3.0])
    
    def _extract_requirement_features(self, requirements: Dict[str, Any]) -> np.ndarray:
        """提取需求特征向量"""
        # 实现需求特征提取逻辑
        return np.array([1.0, 2.0, 3.0])
    
    def _simple_ml_score(self, room_features: np.ndarray, requirement_features: np.ndarray) -> float:
        """简化的ML评分"""
        # 实现简化的ML评分逻辑
        return 0.75
    
    def _calculate_room_type_match(self, room: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """计算房型匹配度"""
        # 实现房型匹配计算
        return 0.8
    
    def _calculate_location_match(self, room: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """计算位置匹配度"""
        # 实现位置匹配计算
        return 0.7
    
    def _calculate_price_match(self, room: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """计算价格匹配度"""
        # 实现价格匹配计算
        return 0.9
    
    def _calculate_facility_match(self, room: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """计算设施匹配度"""
        # 实现设施匹配计算
        return 0.6
    
    def _build_matching_criteria(self, requirements: Dict[str, Any]) -> str:
        """构建匹配标准"""
        # 实现匹配标准构建逻辑
        return "标准匹配标准"
    
    def _parse_llm_evaluation(self, evaluation_result: str) -> Dict[str, Any]:
        """解析LLM评估结果"""
        # 实现LLM结果解析逻辑
        return {'overall_score': 0.75, 'reasoning': '匹配良好'}
    
    def _generate_match_explanation(self, 
                                  scores: Dict[str, float], 
                                  room: Dict[str, Any], 
                                  requirements: Dict[str, Any]) -> str:
        """生成匹配解释"""
        # 实现匹配解释生成逻辑
        return "房源与需求匹配良好"
    
    def _get_recommendation_level(self, score: float) -> str:
        """获取推荐等级"""
        if score >= 0.8:
            return "强烈推荐"
        elif score >= 0.6:
            return "推荐"
        elif score >= 0.4:
            return "一般"
        else:
            return "不推荐"
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """计算分数分布"""
        distribution = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        
        for score in scores:
            if score >= 0.8:
                distribution['excellent'] += 1
            elif score >= 0.6:
                distribution['good'] += 1
            elif score >= 0.4:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def _get_error_result(self, error_message: str) -> Dict[str, Any]:
        """生成错误结果"""
        return {
            'success': False,
            'error_message': error_message,
            'matched_rooms': [],
            'timestamp': datetime.now().isoformat()
        } 