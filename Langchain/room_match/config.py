#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Langchain项目配置文件
包含工作流、Agent、RAG的配置参数
"""

import os
import logging

# ===== 全局配置 =====
PROJECT_ROOT = "/Users/xiemingxuan/B-code/Langchain"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ===== 日志配置 =====
LOG_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': os.path.join(LOGS_DIR, 'langchain_demo.log'),
    'filemode': 'a',
    'encoding': 'utf-8'
}

# ===== 房型匹配配置 =====
ROOM_MATCHING_CONFIG = {
    'matching_method': 'agent',  # 可选: 'llm' 或 'agent'
    'enable_dl_matching': False,  # 是否启用深度学习匹配
    'llm_timeout': 300,  # LLM调用超时时间（秒），从180增加到300
    'agent_max_iterations': 10,  # Agent最大迭代次数，从3增加到10
    'agent_verbose': True,  # Agent是否输出详细信息
    'agent_early_stopping_method': 'generate',  # 早停方法
    'agent_handle_parsing_errors': True,  # 处理解析错误
}

# ===== LLM配置 =====
LLM_CONFIG = { 
    # 阿里云千问配置 (默认)qwen-max deepseek-r1 qwen-plus deepseek-r1-0528
    'qwen': {
        'model_name': 'qwen-plus',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'temperature': 0.7,
        'max_tokens': 20000,  # 从10000增加到20000，支持更长的响应
        'api_key': os.getenv('DASHSCOPE_API_KEY', 'sk-6d6b7b8eac764d08b90a817610f2f6ce')
    }
}
# ===== 工作流配置 =====
WORKFLOW_CONFIG = {
    'max_steps': 10,
    'timeout': 300,  # 5分钟超时
    'retry_count': 3,
    'parallel_execution': False
}

# ===== 示例数据配置 =====
SAMPLE_DATA = {
    'documents': [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "机器学习是人工智能的一个子领域，专注于让计算机系统自动从数据中学习和改进，而无需明确编程。",
        "深度学习是机器学习的一个子集，使用人工神经网络来模拟人脑的学习过程。",
        "自然语言处理是人工智能的一个分支，专注于让计算机理解、解释和生成人类语言。"
    ],
    'questions': [
        "什么是人工智能？",
        "机器学习和深度学习有什么区别？",
        "自然语言处理的主要应用是什么？"
    ]
}

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(**LOG_CONFIG)
    return logging.getLogger(__name__)

def get_qwen_client():
    """创建并返回一个阿里云千问客户端实例"""
    from openai import OpenAI
    qwen_config = LLM_CONFIG['qwen']
    
    # 设置环境变量
    if os.getenv("DASHSCOPE_API_KEY") is None:
        os.environ["DASHSCOPE_API_KEY"] = qwen_config['api_key']
    
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=qwen_config['base_url'],
    )

def get_config(config_type):
    """获取指定类型的配置"""
    configs = {
        'llm': LLM_CONFIG,
        'workflow': WORKFLOW_CONFIG
    }
    return configs.get(config_type, {})

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("配置加载完成")
    print("Langchain项目配置已加载完成！") 