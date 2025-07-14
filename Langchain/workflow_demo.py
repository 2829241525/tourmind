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

class ContentAnalysisResult(BaseModel):
    """内容分析结果模型"""
    sentiment: str = Field(description="情感分析结果: 正面、负面或中性")
    keywords: List[str] = Field(description="关键词列表")
    summary: str = Field(description="内容摘要")
    category: str = Field(description="内容分类")

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
            elif self.llm_type == 'ollama':
                # Ollama模型配置
                from langchain_community.llms import Ollama
                self.llm = Ollama(
                    model=self.llm_config['model_name'],
                    base_url=self.llm_config['base_url'],
                    temperature=self.llm_config['temperature']
                )
            else:
                # 默认OpenAI配置
                self.llm = ChatOpenAI(
                    model_name=self.llm_config['model_name'],
                    temperature=self.llm_config['temperature'],
                    max_tokens=self.llm_config['max_tokens'],
                    openai_api_key=self.llm_config['api_key']
                )
            
            self.logger.info(f"语言模型设置完成 - model: {self.llm_config['model_name']}")
            
        except Exception as e:
            self.logger.error(f"组件设置失败: {str(e)}")
            raise
    
    def create_simple_sequential_workflow(self) -> SimpleSequentialChain:
        """创建简单顺序工作流"""
        # 第一步：内容分析
        analysis_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            请分析以下文本内容：
            
            文本: {text}
            
            请提供：
            1. 情感分析（正面/负面/中性）
            2. 主要主题
            3. 简要摘要
            
            分析结果：
            """
        )
        
        analysis_chain = LLMChain(
            llm=self.llm,
            prompt=analysis_prompt,
            output_key="analysis"
        )
        
        # 第二步：建议生成
        suggestion_prompt = PromptTemplate(
            input_variables=["analysis"],
            template="""
            基于以下内容分析结果，提供改进建议：
            
            分析结果: {analysis}
            
            请提供具体的改进建议和行动方案：
            """
        )
        
        suggestion_chain = LLMChain(
            llm=self.llm,
            prompt=suggestion_prompt,
            output_key="suggestions"
        )
        
        # 创建顺序链
        workflow = SimpleSequentialChain(
            chains=[analysis_chain, suggestion_chain],
            verbose=True
        )
        
        return workflow
    
    def create_complex_sequential_workflow(self) -> SequentialChain:
        """创建复杂顺序工作流"""
        # 步骤1：内容预处理
        preprocessing_prompt = PromptTemplate(
            input_variables=["raw_text"],
            template="""
            请对以下原始文本进行预处理，包括：
            1. 去除多余空格和换行
            2. 纠正明显的拼写错误
            3. 统一格式
            
            原始文本: {raw_text}
            
            预处理后的文本：
            """
        )
        
        preprocessing_chain = LLMChain(
            llm=self.llm,
            prompt=preprocessing_prompt,
            output_key="cleaned_text"
        )
        
        # 步骤2：关键词提取
        keyword_prompt = PromptTemplate(
            input_variables=["cleaned_text"],
            template="""
            从以下文本中提取5-8个最重要的关键词，用逗号分隔：
            
            文本: {cleaned_text}
            
            关键词：
            """
        )
        
        keyword_chain = LLMChain(
            llm=self.llm,
            prompt=keyword_prompt,
            output_key="keywords"
        )
        
        # 步骤3：情感分析
        sentiment_prompt = PromptTemplate(
            input_variables=["cleaned_text"],
            template="""
            分析以下文本的情感倾向，只回答：正面、负面或中性
            
            文本: {cleaned_text}
            
            情感倾向：
            """
        )
        
        sentiment_chain = LLMChain(
            llm=self.llm,
            prompt=sentiment_prompt,
            output_key="sentiment"
        )
        
        # 步骤4：摘要生成
        summary_prompt = PromptTemplate(
            input_variables=["cleaned_text"],
            template="""
            为以下文本生成一个简洁的摘要（不超过100字）：
            
            文本: {cleaned_text}
            
            摘要：
            """
        )
        
        summary_chain = LLMChain(
            llm=self.llm,
            prompt=summary_prompt,
            output_key="summary"
        )
        
        # 步骤5：最终报告
        report_prompt = PromptTemplate(
            input_variables=["cleaned_text", "keywords", "sentiment", "summary"],
            template="""
            基于以下信息生成完整的内容分析报告：
            
            原始文本: {cleaned_text}
            关键词: {keywords}
            情感倾向: {sentiment}
            摘要: {summary}
            
            完整分析报告：
            """
        )
        
        report_chain = LLMChain(
            llm=self.llm,
            prompt=report_prompt,
            output_key="final_report"
        )
        
        # 创建复杂顺序链
        workflow = SequentialChain(
            chains=[preprocessing_chain, keyword_chain, sentiment_chain, summary_chain, report_chain],
            input_variables=["raw_text"],
            output_variables=["cleaned_text", "keywords", "sentiment", "summary", "final_report"],
            verbose=True
        )
        
        return workflow
    
    def create_custom_workflow(self) -> List[WorkflowStep]:
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

def demo_simple_workflow(llm_type='qwen'):
    """简单工作流演示"""
    print(f"🔄 简单顺序工作流演示 - 使用模型: {llm_type}")
    print("-" * 30)
    
    try:
        workflow_demo = WorkflowDemo(llm_type=llm_type)
        simple_workflow = workflow_demo.create_simple_sequential_workflow()
        
        test_text = "这个产品的质量很好，我非常满意。但是价格有点贵，希望能有更多优惠。"
        result = simple_workflow.run(test_text)
        
        print(f"输入文本: {test_text}")
        print(f"工作流结果: {result}")
        
    except Exception as e:
        print(f"❌ 简单工作流演示失败: {str(e)}")

def demo_complex_workflow(llm_type='qwen'):
    """复杂工作流演示"""
    print(f"\n🔄 复杂顺序工作流演示 - 使用模型: {llm_type}")
    print("-" * 30)
    
    try:
        workflow_demo = WorkflowDemo(llm_type=llm_type)
        complex_workflow = workflow_demo.create_complex_sequential_workflow()
        
        test_text = """
        人工智能    技术   发展很快，   特别是在   自然语言处理领域。
        LangChain 是一个 很棒的框架，  可以帮助开发者构建  AI应用。
        但是学习曲线  有点陡峭，  需要一定的技术基础。
        """
        
        result = complex_workflow({"raw_text": test_text})
        
        print(f"输入文本: {test_text}")
        print(f"处理结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ 复杂工作流演示失败: {str(e)}")

def demo_custom_workflow(llm_type='qwen'):
    """自定义工作流演示"""
    print(f"\n🔄 自定义工作流演示 - 使用模型: {llm_type}")
    print("-" * 30)
    
    try:
        workflow_demo = WorkflowDemo(llm_type=llm_type)
        custom_steps = workflow_demo.create_custom_workflow()
        
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
    
    # 演示简单工作流
    demo_simple_workflow(llm_type)
    
    # 演示复杂工作流
    demo_complex_workflow(llm_type)
    
    # 演示自定义工作流
    demo_custom_workflow(llm_type)
    
    print("\n✅ 工作流演示完成!")

if __name__ == "__main__":
    demo_workflow() 