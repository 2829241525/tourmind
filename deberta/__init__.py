#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deberta房型匹配模块
包含基于DeBERTa模型的房型匹配相关功能
"""

import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 版本信息
__version__ = '1.0.0' 