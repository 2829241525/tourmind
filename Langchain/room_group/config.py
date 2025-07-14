#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房间分组项目配置文件
"""

import os

# ===== LLM配置 =====
LLM_CONFIG = { 
    # 阿里云千问配置 (默认)
    'qwen': {
        'model_name': 'qwen-plus',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'temperature': 0.1,
        'max_tokens': 500,
        'api_key': os.getenv('DASHSCOPE_API_KEY', 'sk-6d6b7b8eac764d08b90a817610f2f6ce')
    }
} 