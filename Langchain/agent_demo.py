#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent 演示
展示如何使用Langchain构建智能代理系统
"""

import os
import sys
import math
from typing import List, Dict, Any, Optional
import logging

# 导入配置
from config import AGENT_CONFIG, LLM_CONFIG, setup_logging

try:
    from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
    from langchain_openai import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.tools import BaseTool
    from langchain.callbacks.manager import CallbackManagerForToolRun
    from langchain.schema import AgentAction, AgentFinish
except ImportError as e:
    print(f"请安装required packages: pip install langchain langchain-openai")
    sys.exit(1)

class CalculatorTool(BaseTool):
    """计算器工具"""
    # name: str = "calculator"
    # description: str = "用于执行数学计算。输入应该是一个数学表达式，例如: 2+2 或 sqrt(16)"
    name: str = "search"
    description: str = "用于搜索信息。输入应该是搜索查询"
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """执行计算"""
        try:
            # 安全的数学表达式评估
            allowed_names = {
                k: v for k, v in math.__dict__.items() 
                if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round})
            
            result = eval(query, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"计算错误: {str(e)}"

class FileReaderTool(BaseTool):
    """文件读取工具"""
    name: str = "file_reader"
    description: str = "用于读取文件内容。输入应该是文件路径"
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """读取文件"""
        try:
            if not os.path.exists(query):
                return f"文件不存在: {query}"
            
            with open(query, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 限制返回内容长度
            if len(content) > 1000:
                content = content[:1000] + "...(文件内容过长，已截断)"
            
            return content
        except Exception as e:
            return f"文件读取错误: {str(e)}"

class SearchTool(BaseTool):
    """搜索工具（模拟）"""
    # name: str = "search"
    # description: str = "用于搜索信息。输入应该是搜索查询"
    name: str = "calculator"
    description: str = "用于执行数学计算。输入应该是一个数学表达式，例如: 2+2 或 sqrt(16)"
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """执行搜索（模拟）"""
        # 这里是模拟的搜索结果
        search_results = {
            "人工智能": "人工智能(AI)是指由人制造出来的机器所表现出来的智能。通常人工智能是指通过普通计算机程序来呈现人类智能的技术。",
            "机器学习": "机器学习是人工智能的一个分支，专注于算法的使用，这些算法可以通过经验自动改进。",
            "深度学习": "深度学习是机器学习的子集，它使用具有多层的神经网络来模拟和理解复杂模式。",
            "langchain": "LangChain是一个开源框架，用于开发由语言模型驱动的应用程序。",
            "python": "Python是一种高级编程语言，以其清晰的语法和强大的功能而闻名。"
        }
        
        # 查找最相关的结果
        for key, value in search_results.items():
            if key.lower() in query.lower():
                return f"搜索结果: {value}"
        
        return f"未找到关于'{query}'的相关信息"

class AgentDemo:
    """Agent演示类"""
    
    def __init__(self, llm_type='qwen'):
        """初始化Agent系统
        
        Args:
            llm_type: 语言模型类型，支持 'qwen', 'openai', 'ollama'
        """
        self.logger = setup_logging()
        self.config = AGENT_CONFIG
        self.llm_type = llm_type
        self.llm_config = LLM_CONFIG.get(llm_type, LLM_CONFIG['qwen'])
        
        # 初始化组件
        self.llm = None
        self.tools = []
        self.memory = None
        self.agent_executor = None
        
        self.logger.info(f"Agent系统初始化开始 - 使用模型: {llm_type}")
        self._setup_components()
        self.logger.info("Agent系统初始化完成")
    
    def _setup_components(self):
        """设置Agent组件"""
        try:

            self.llm = ChatOpenAI(
                    model_name=self.llm_config['model_name'],
                    temperature=self.llm_config['temperature'],
                    max_tokens=self.llm_config['max_tokens'],
                    openai_api_key=self.llm_config['api_key'],
                    openai_api_base=self.llm_config['base_url']
                )
            
            self.logger.info(f"语言模型设置完成 - model: {self.llm_config['model_name']}")
            
            # 2. 工具
            self._setup_tools()
            
            # 3. 记忆
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            self.logger.info("对话记忆设置完成")
            
            # 4. Agent执行器
            self._setup_agent()
            
        except Exception as e:
            self.logger.error(f"组件设置失败: {str(e)}")
            raise
    
    def _setup_tools(self):
        """设置工具"""
        available_tools = {
            'calculator': CalculatorTool(),
            'file_reader': FileReaderTool(),
            'search': SearchTool()
        }
        
        # 根据配置启用工具
        self.tools = []
        for tool_name in self.config['tools_enabled']:
            if tool_name in available_tools:
                self.tools.append(available_tools[tool_name])
                self.logger.info(f"工具启用: {tool_name}")
        
        if not self.tools:
            self.logger.warning("没有启用任何工具")
    
    def _setup_agent(self):
        """设置Agent"""
        try:
            self.agent_executor = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=self.config['verbose'],
                max_iterations=self.config['max_iterations'],
                handle_parsing_errors=True
            )
            self.logger.info("Agent执行器设置完成")
            
        except Exception as e:
            self.logger.error(f"Agent设置失败: {str(e)}")
            raise
    
    def run(self, query: str) -> Dict[str, Any]:
        """运行Agent处理查询"""
        try:
            self.logger.info(f"收到查询: {query}")
            
            # 执行查询
            result = self.agent_executor.run(query)
            
            response = {
                "query": query,
                "result": result,
                "success": True
            }
            
            self.logger.info("查询处理完成")
            return response
            
        except Exception as e:
            self.logger.error(f"查询处理失败: {str(e)}")
            return {
                "query": query,
                "result": f"处理失败: {str(e)}",
                "success": False
            }
    
    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return [tool.name for tool in self.tools]
    
    def clear_memory(self):
        """清空对话记忆"""
        self.memory.clear()
        self.logger.info("对话记忆已清空")

def demo_agent(llm_type='qwen'):
    """Agent演示函数
    
    Args:
        llm_type: 语言模型类型，支持 'qwen', 'openai', 'ollama'
    """
    print(f"🤖 Agent (智能代理) 演示开始 - 使用模型: {llm_type}")
    print("=" * 50)
    
    try:
        # 初始化Agent系统
        agent = AgentDemo(llm_type=llm_type)
        
        # 显示可用工具
        tools = agent.get_available_tools()
        print(f"🛠️  可用工具: {', '.join(tools)}")
        
        # 演示查询
        demo_queries = [
            "计算 (25 + 15) * 2 的结果",
            "搜索关于人工智能的信息",
            "什么是机器学习？请搜索相关信息",
            "计算圆周率的近似值: 22/7",
            "搜索Python编程语言的信息"
        ]
        
        print("\n📝 Agent演示查询:")
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{i}. 查询: {query}")
            result = agent.run(query)
            
            if result['success']:
                print(f"   回答: {result['result']}")
            else:
                print(f"   ❌ 错误: {result['result']}")
        
        # 演示对话记忆
        print("\n🧠 对话记忆演示:")
        memory_queries = [
            "记住：我的名字是张三",
            "我刚才告诉你我的名字是什么？"
        ]
        
        for query in memory_queries:
            print(f"\n查询: {query}")
            result = agent.run(query)
            print(f"回答: {result['result']}")
        
        print("\n✅ Agent演示完成!")
        
    except Exception as e:
        print(f"❌ Agent演示失败: {str(e)}")
        print("请检查配置和API密钥")

if __name__ == "__main__":
    demo_agent() 