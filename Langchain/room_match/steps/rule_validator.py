#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤2：规则校验器
校验大模型抽取的规则和格式，如果格式存在问题，再次访问大模型进行整理
"""

import json
import logging
import sys
import os
from typing import Dict, Any, Optional, List

# 添加父目录到路径以支持直接运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import LLM_CONFIG
    from prompts import get_prompt
except ImportError:
    # 如果作为模块导入时，使用相对导入
    from ..config import LLM_CONFIG
    from ..prompts import get_prompt

logger = logging.getLogger(__name__)

class RuleValidator:
    """规则校验器 - 步骤2"""
    
    def __init__(self, llm_client=None, **kwargs):
        """
        初始化规则校验器
        
        Args:
            llm_client: 大模型客户端
            **kwargs: 其他配置参数
        """
        self.llm_client = llm_client
        self.config = kwargs
        
    def validate_extraction_rules(self,
                                extraction_rules: Dict[str, Any],
                                validation_schema: Optional[Dict] = None,
                                **kwargs) -> Dict[str, Any]:
        """
        校验抽取规则的格式和完整性
        
        Args:
            extraction_rules: 待校验的抽取规则
            validation_schema: 校验模式
            **kwargs: 其他校验参数
            
        Returns:
            校验结果
        """
        try:
            # TODO: 实现规则校验逻辑
            validation_errors = []
            validation_warnings = []
            
            # 框架示例
            if not extraction_rules:
                validation_errors.append("抽取规则为空")
            
            # 检查必要字段
            required_fields = ["field_mapping", "format_conversion", "validation_rules"]
            for field in required_fields:
                if field not in extraction_rules:
                    validation_errors.append(f"缺少必要字段: {field}")
            
            validation_result = {
                "is_valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "warnings": validation_warnings,
                "details": {}
            }
            
            logger.info(f"规则校验完成，发现{len(validation_errors)}个错误")
            return validation_result
            
        except Exception as e:
            logger.error(f"规则校验失败: {str(e)}")
            return {
                "is_valid": False,
                "errors": [str(e)],
                "warnings": [],
                "details": {}
            }
    
    def format_validation_check(self,
                              extraction_rules: Dict[str, Any],
                              expected_format: str = "json",
                              **kwargs) -> Dict[str, Any]:
        """
        检查规则格式是否符合要求
        
        Args:
            extraction_rules: 抽取规则
            expected_format: 期望格式
            **kwargs: 其他参数
            
        Returns:
            格式检查结果
        """
        try:
            # TODO: 实现格式检查逻辑
            format_errors = []
            
            # 检查JSON格式
            try:
                json.dumps(extraction_rules)
            except Exception as e:
                format_errors.append(f"JSON格式错误: {str(e)}")
            
            format_result = {
                "format_valid": len(format_errors) == 0,
                "format_errors": format_errors,
                "current_format": type(extraction_rules).__name__
            }
            
            logger.info("格式检查完成")
            return format_result
            
        except Exception as e:
            logger.error(f"格式检查失败: {str(e)}")
            return {
                "format_valid": False,
                "format_errors": [str(e)],
                "current_format": "unknown"
            }
    
    def fix_rules_with_llm(self,
                          extraction_rules: Dict[str, Any],
                          validation_errors: List[str],
                          data_samples: Optional[Dict] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        使用大模型修正抽取规则
        
        Args:
            extraction_rules: 原始抽取规则
            validation_errors: 校验发现的错误
            data_samples: 数据样本用于参考
            **kwargs: 其他参数
            
        Returns:
            修正后的抽取规则
        """
        try:
            # 获取提示词模板
            prompt_template = get_prompt('rule_validation')
            
            # 构建提示词
            prompt_input = {
                "extraction_rules": json.dumps(extraction_rules, ensure_ascii=False, indent=2),
                "validation_errors": "\n".join(validation_errors),
                "data_samples": json.dumps(data_samples or {}, ensure_ascii=False, indent=2)
            }
            
            # TODO: 调用大模型修正规则
            # prompt = prompt_template.format(**prompt_input)
            # llm_response = self.llm_client.generate(prompt)
            # fixed_rules = self._parse_llm_response(llm_response)
            
            # 框架示例
            fixed_rules = {
                "field_mapping": extraction_rules.get("field_mapping", {}),
                "format_conversion": extraction_rules.get("format_conversion", {}),
                "validation_rules": extraction_rules.get("validation_rules", {}),
                "cleaning_rules": extraction_rules.get("cleaning_rules", {}),
                "metadata": {
                    "fixed_at": "",
                    "original_errors": validation_errors,
                    "version": "fixed_v1"
                }
            }
            
            logger.info("规则修正完成")
            return fixed_rules
            
        except Exception as e:
            logger.error(f"规则修正失败: {str(e)}")
            return {"error": str(e)}
    
    def execute_step(self,
                    step1_result: Dict[str, Any],
                    validation_schema: Optional[Dict] = None,
                    max_retry_count: int = 3,
                    **kwargs) -> Dict[str, Any]:
        """
        执行完整的步骤2流程
        
        Args:
            step1_result: 步骤1的执行结果
            validation_schema: 校验模式
            max_retry_count: 最大重试次数
            **kwargs: 其他参数
            
        Returns:
            步骤2执行结果
        """
        try:
            extraction_rules = step1_result.get("extraction_rules", {})
            retry_count = 0
            
            while retry_count < max_retry_count:
                # 1. 校验抽取规则
                validation_result = self.validate_extraction_rules(
                    extraction_rules, 
                    validation_schema, 
                    **kwargs
                )
                
                # 2. 检查格式
                format_result = self.format_validation_check(
                    extraction_rules, 
                    **kwargs
                )
                
                # 如果校验通过，直接返回
                if validation_result["is_valid"] and format_result["format_valid"]:
                    result = {
                        "step": 2,
                        "status": "success",
                        "validated_rules": extraction_rules,
                        "validation_result": validation_result,
                        "format_result": format_result,
                        "retry_count": retry_count,
                        "metadata": {
                            "final_validation": True,
                            "errors_fixed": retry_count > 0
                        }
                    }
                    
                    logger.info(f"步骤2执行完成，重试{retry_count}次")
                    return result
                
                # 3. 如果校验失败，使用大模型修正
                all_errors = validation_result["errors"] + format_result["format_errors"]
                data_samples = step1_result.get("formatted_data", {})
                
                extraction_rules = self.fix_rules_with_llm(
                    extraction_rules,
                    all_errors,
                    data_samples,
                    **kwargs
                )
                
                retry_count += 1
                logger.warning(f"规则校验失败，第{retry_count}次重试修正")
            
            # 达到最大重试次数
            result = {
                "step": 2,
                "status": "partial_success",
                "validated_rules": extraction_rules,
                "validation_result": validation_result,
                "format_result": format_result,
                "retry_count": retry_count,
                "warning": f"达到最大重试次数{max_retry_count}，规则可能仍有问题"
            }
            
            logger.warning("步骤2部分完成，规则可能仍有问题")
            return result
            
        except Exception as e:
            logger.error(f"步骤2执行失败: {str(e)}")
            return {
                "step": 2,
                "status": "error",
                "error": str(e)
            }
    
    def _parse_llm_response(self, llm_response):
        """解析大模型响应"""
        # TODO: 实现大模型响应解析逻辑
        pass 