#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow 演示
展示如何使用Langchain构建工作流系统
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional, Callable
import logging
from datetime import datetime

# 导入配置
from config import WORKFLOW_CONFIG, LLM_CONFIG, setup_logging

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
    from langchain.schema import BaseOutputParser
    from langchain.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"请安装required packages: pip install langchain langchain-openai pydantic")
    sys.exit(1)

class WorkflowStep:
    """工作流步骤类"""
    
    def __init__(self, name: str, description: str, func: Callable, inputs: List[str] = None, outputs: List[str] = None):
        self.name = name
        self.description = description
        self.func = func
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.execution_time = 0
        self.success = False
        self.error_message = ""
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流步骤"""
        start_time = time.time()
        try:
            result = self.func(context)
            self.success = True
            self.execution_time = time.time() - start_time
            return result
        except Exception as e:
            self.success = False
            self.error_message = str(e)
            self.execution_time = time.time() - start_time
            raise



class WorkflowDemo:
    """工作流演示类"""
    
    def __init__(self, llm_type='qwen'):
        """初始化工作流系统
        
        Args:
            llm_type: 语言模型类型，支持 'qwen', 'openai', 'ollama'
        """
        self.logger = setup_logging()
        self.config = WORKFLOW_CONFIG
        self.llm_type = llm_type
        self.llm_config = LLM_CONFIG.get(llm_type, LLM_CONFIG['qwen'])
        
        # 初始化组件
        self.llm = None
        self.steps = []
        self.context = {}
        
        self.logger.info(f"工作流系统初始化开始 - 使用模型: {llm_type}")
        self._setup_components()
        self.logger.info("工作流系统初始化完成")
    
    def _setup_components(self):
        """设置工作流组件"""
        try:
            # 语言模型
            if self.llm_type == 'qwen':
                # 千问模型配置
                self.llm = ChatOpenAI(
                    model_name=self.llm_config['model_name'],
                    temperature=self.llm_config['temperature'],
                    max_tokens=self.llm_config['max_tokens'],
                    openai_api_key=self.llm_config['api_key'],
                    openai_api_base=self.llm_config['base_url']
                )
            
            self.logger.info(f"语言模型设置完成 - model: {self.llm_config['model_name']}")
            
        except Exception as e:
            self.logger.error(f"组件设置失败: {str(e)}")
            raise
   
    
    def room_match_workflow(self) -> List[WorkflowStep]:
        """创建自定义工作流"""
        def step1_input_validation(context):
            """步骤1：输入验证"""
            text = context.get('input_text', '')
            if not text or len(text.strip()) < 10:
                raise ValueError("输入文本太短，至少需要10个字符")
            
            context['validated_text'] = text.strip()
            return context
        
        def step2_text_analysis(context):
            """步骤2：文本分析"""
            text = context['validated_text']
            
            # 简单的文本统计
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            
            context['text_stats'] = {
                'word_count': word_count,
                'char_count': char_count,
                'sentence_count': sentence_count
            }
            return context
        
        def step3_llm_analysis(context):
            """步骤3：LLM分析"""
            text = context['validated_text']
            
            prompt = PromptTemplate(
                input_variables=["text"],
                template="分析以下文本的主题和情感：\n\n{text}\n\n分析结果："
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = chain.run(text=text)
            
            context['llm_analysis'] = result
            return context
        
        def step4_result_formatting(context):
            """步骤4：结果格式化"""
            stats = context['text_stats']
            analysis = context['llm_analysis']
            
            formatted_result = {
                'timestamp': datetime.now().isoformat(),
                'input_text': context['validated_text'],
                'statistics': stats,
                'ai_analysis': analysis,
                'processing_completed': True
            }
            
            context['final_result'] = formatted_result
            return context
        
        # 创建工作流步骤
        steps = [
            WorkflowStep("输入验证", "验证输入文本的有效性", step1_input_validation),
            WorkflowStep("文本统计", "计算基本文本统计信息", step2_text_analysis),
            WorkflowStep("AI分析", "使用LLM进行深度分析", step3_llm_analysis),
            WorkflowStep("结果格式化", "格式化最终输出结果", step4_result_formatting)
        ]
        
        return steps
    
    def execute_custom_workflow(self, steps: List[WorkflowStep], initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """执行自定义工作流"""
        context = initial_context.copy()
        execution_log = []
        
        self.logger.info(f"开始执行工作流，共{len(steps)}个步骤")
        
        for i, step in enumerate(steps, 1):
            try:
                self.logger.info(f"执行步骤 {i}/{len(steps)}: {step.name}")
                print(f"🔄 执行步骤 {i}: {step.name}")
                
                start_time = time.time()
                context = step.execute(context)
                execution_time = time.time() - start_time
                
                log_entry = {
                    'step': i,
                    'name': step.name,
                    'description': step.description,
                    'success': True,
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                execution_log.append(log_entry)
                self.logger.info(f"步骤 {i} 执行完成，耗时 {execution_time:.2f}秒")
                
            except Exception as e:
                error_msg = f"步骤 {i} 执行失败: {str(e)}"
                self.logger.error(error_msg)
                
                log_entry = {
                    'step': i,
                    'name': step.name,
                    'description': step.description,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                
                execution_log.append(log_entry)
                raise Exception(error_msg)
        
        context['execution_log'] = execution_log
        self.logger.info("工作流执行完成")
        return context

def demo_custom_workflow(llm_type='qwen'):
    """自定义工作流演示"""
    print(f"\n🔄 自定义工作流演示 - 使用模型: {llm_type}")
    print("-" * 30)
    
    try:
        workflow_demo = WorkflowDemo(llm_type=llm_type)
        custom_steps = workflow_demo.room_match_workflow()
        
        initial_context = {
            'input_text': "LangChain是一个强大的框架，用于构建基于大语言模型的应用程序。它提供了丰富的工具和组件，帮助开发者快速构建智能应用。"
        }
        
        result = workflow_demo.execute_custom_workflow(custom_steps, initial_context)
        
        print("✅ 自定义工作流执行完成")
        print(f"最终结果: {json.dumps(result['final_result'], ensure_ascii=False, indent=2)}")
        
        print("\n📊 执行日志:")
        for log in result['execution_log']:
            status = "✅" if log['success'] else "❌"
            print(f"  {status} 步骤 {log['step']}: {log['name']}")
            if 'execution_time' in log:
                print(f"     耗时: {log['execution_time']:.2f}秒")
        
    except Exception as e:
        print(f"❌ 自定义工作流演示失败: {str(e)}")

def demo_workflow(llm_type='qwen'):
    """工作流演示主函数
    
    Args:
        llm_type: 语言模型类型，支持 'qwen', 'openai', 'ollama'
    """
    print(f"🚀 Workflow (工作流) 演示开始 - 使用模型: {llm_type}")
    print("=" * 50)

    # 演示自定义工作流
    demo_custom_workflow(llm_type)
    
    print("\n✅ 工作流演示完成!")

if __name__ == "__main__":
    demo_workflow() 