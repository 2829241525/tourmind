#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房型匹配工作流管理器
协调管理4个步骤的执行流程
"""

import logging
import sys
import os
from typing import Dict, Any, Optional, List

# 添加父目录到路径以支持直接运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import LLM_CONFIG
    from steps import (
        HttpInfoProcessor,
        RuleValidator,
        RoomMatchingEngine,
        StandardRoomGenerator
    )
except ImportError:
    # 如果作为模块导入时，使用相对导入
    from .config import LLM_CONFIG
    from .steps import (
        HttpInfoProcessor,
        RuleValidator,
        RoomMatchingEngine,
        StandardRoomGenerator
    )

logger = logging.getLogger(__name__)

class RoomMatchingWorkflow:
    """房型匹配工作流管理器"""
    
    def __init__(self, llm_client=None, **kwargs):
        """
        初始化工作流管理器
        
        Args:
            llm_client: 大模型客户端
            **kwargs: 配置参数
        """
        self.llm_client = llm_client
        self.config = kwargs
        
        # 初始化各步骤处理器
        self.step1_processor = HttpInfoProcessor(llm_client, **kwargs)
        self.step2_validator = RuleValidator(llm_client, **kwargs)
        self.step3_engine = RoomMatchingEngine(llm_client, **kwargs)
        self.step4_generator = StandardRoomGenerator(llm_client, **kwargs)
         
        # 工作流状态
        self.workflow_state = {
            "current_step": 0,
            "step_results": {},
            "workflow_id": "",
            "status": "initialized"
        }
    
    def execute_full_workflow(self,
                            url: str,
                            extraction_requirements: str,
                            room_data: List[Dict[str, Any]],
                            matching_criteria: Dict[str, Any],
                            generation_rules: Dict[str, Any],
                            **kwargs) -> Dict[str, Any]:
        """
        执行完整的4步工作流
        
        Args:
            url: HTTP请求URL
            extraction_requirements: 信息抽取要求
            room_data: 房型数据列表
            matching_criteria: 匹配标准
            generation_rules: 标准房型生成规则
            **kwargs: 其他参数
            
        Returns:
            完整工作流执行结果
        """
        try:
            logger.info("开始执行完整房型匹配工作流")
            
            # 步骤1: HTTP信息处理和规则生成
            logger.info("执行步骤1: HTTP信息处理")
            step1_result = self.step1_processor.execute_step(
                url=url,
                extraction_requirements=extraction_requirements,
                **kwargs
            )
            self.workflow_state["step_results"]["step1"] = step1_result
            
            if step1_result.get("status") != "success":
                return self._create_error_result("步骤1执行失败", step1_result)
            
            # 步骤2: 规则校验和修正
            logger.info("执行步骤2: 规则校验")
            step2_result = self.step2_validator.execute_step(
                step1_result=step1_result,
                **kwargs
            )
            self.workflow_state["step_results"]["step2"] = step2_result
            
            if step2_result.get("status") == "error":
                return self._create_error_result("步骤2执行失败", step2_result)
            
            # 步骤3: 房型匹配
            logger.info("执行步骤3: 房型匹配")
            step3_result = self.step3_engine.execute_step(
                step2_result=step2_result,
                room_data=room_data,
                matching_criteria=matching_criteria,
                **kwargs
            )
            self.workflow_state["step_results"]["step3"] = step3_result
            
            if step3_result.get("status") != "success":
                return self._create_error_result("步骤3执行失败", step3_result)
            
            # 步骤4: 标准房型生成
            logger.info("执行步骤4: 标准房型生成")
            step4_result = self.step4_generator.execute_step(
                step3_result=step3_result,
                generation_rules=generation_rules,
                **kwargs
            )
            self.workflow_state["step_results"]["step4"] = step4_result
            
            # 生成最终结果
            final_result = {
                "workflow_status": "completed",
                "step1_result": step1_result,
                "step2_result": step2_result,
                "step3_result": step3_result,
                "step4_result": step4_result,
                "summary": self._generate_workflow_summary(),
                "metadata": {
                    "total_rooms": len(room_data),
                    "matched_count": step3_result.get("metadata", {}).get("matched_count", 0),
                    "unmatched_count": step3_result.get("metadata", {}).get("unmatched_count", 0),
                    "new_standards_generated": step4_result.get("metadata", {}).get("generated_count", 0)
                }
            }
            
            self.workflow_state["status"] = "completed"
            logger.info("完整工作流执行完成")
            return final_result
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            return self._create_error_result("工作流执行异常", {"error": str(e)})
    
    def execute_partial_workflow(self,
                                start_step: int,
                                end_step: int,
                                **kwargs) -> Dict[str, Any]:
        """
        执行部分工作流
        
        Args:
            start_step: 起始步骤 (1-4)
            end_step: 结束步骤 (1-4)
            **kwargs: 步骤参数
            
        Returns:
            部分工作流执行结果
        """
        try:
            logger.info(f"执行部分工作流: 步骤{start_step}到步骤{end_step}")
            
            # TODO: 实现部分工作流执行逻辑
            result = {
                "workflow_status": "partial_completed",
                "start_step": start_step,
                "end_step": end_step,
                "executed_steps": [],
                "results": {}
            }
            
            logger.info("部分工作流执行完成")
            return result
            
        except Exception as e:
            logger.error(f"部分工作流执行失败: {str(e)}")
            return self._create_error_result("部分工作流执行异常", {"error": str(e)})
    
    def execute_single_step(self,
                          step_number: int,
                          **kwargs) -> Dict[str, Any]:
        """
        执行单个步骤
        
        Args:
            step_number: 步骤编号 (1-4)
            **kwargs: 步骤参数
            
        Returns:
            单步执行结果
        """
        try:
            logger.info(f"执行单个步骤: 步骤{step_number}")
            
            if step_number == 1:
                result = self.step1_processor.execute_step(**kwargs)
            elif step_number == 2:
                result = self.step2_validator.execute_step(**kwargs)
            elif step_number == 3:
                result = self.step3_engine.execute_step(**kwargs)
            elif step_number == 4:
                result = self.step4_generator.execute_step(**kwargs)
            else:
                return self._create_error_result("无效步骤编号", {"step_number": step_number})
            
            logger.info(f"步骤{step_number}执行完成")
            return result
            
        except Exception as e:
            logger.error(f"步骤{step_number}执行失败: {str(e)}")
            return self._create_error_result(f"步骤{step_number}执行异常", {"error": str(e)})
    
    def get_workflow_state(self) -> Dict[str, Any]:
        """获取当前工作流状态"""
        return self.workflow_state.copy()
    
    def reset_workflow(self):
        """重置工作流状态"""
        self.workflow_state = {
            "current_step": 0,
            "step_results": {},
            "workflow_id": "",
            "status": "initialized"
        }
        logger.info("工作流状态已重置")
    
    def _generate_workflow_summary(self) -> Dict[str, Any]:
        """生成工作流执行摘要"""
        # TODO: 实现摘要生成逻辑
        return {
            "execution_summary": "工作流执行完成",
            "key_metrics": {},
            "recommendations": []
        }
    
    def _create_error_result(self, error_message: str, error_details: Dict) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            "workflow_status": "error",
            "error_message": error_message,
            "error_details": error_details,
            "workflow_state": self.workflow_state
        }

# ===== 便捷函数 =====
def create_room_matching_workflow(llm_client=None, **kwargs) -> RoomMatchingWorkflow:
    """创建房型匹配工作流实例"""
    return RoomMatchingWorkflow(llm_client, **kwargs)

def quick_room_match(url: str,
                    extraction_requirements: str,
                    room_data: List[Dict[str, Any]],
                    matching_criteria: Dict[str, Any],
                    generation_rules: Dict[str, Any],
                    llm_client=None,
                    **kwargs) -> Dict[str, Any]:
    """快速执行房型匹配工作流"""
    workflow = create_room_matching_workflow(llm_client, **kwargs)
    return workflow.execute_full_workflow(
        url=url,
        extraction_requirements=extraction_requirements,
        room_data=room_data,
        matching_criteria=matching_criteria,
        generation_rules=generation_rules,
        **kwargs
    ) 