#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
房型匹配模型测试脚本
用于验证模型加载和预测功能
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# 使用绝对导入
from deberta.inference import SimCSEPredictor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_loading():
    """测试模型加载"""
    model_path = "/home/maxon/disk2/roomMatch/room_match/deberta/checkpoints_cross_entropy/best_model"
    
    try:
        logger.info(f"尝试加载模型: {model_path}")
        predictor = SimCSEPredictor(model_path)
        logger.info("模型加载成功!")
        
        # 打印模型信息
        if hasattr(predictor, 'loss_type'):
            logger.info(f"模型损失类型: {predictor.loss_type}")
        
        if hasattr(predictor.model, 'config'):
            logger.info(f"模型类型: {predictor.model.config.model_type}")
            
        # 检查特定方法是否存在
        methods = ['encode_batch', 'encode_text_pair', 'calculate_similarity']
        for method in methods:
            if hasattr(predictor, method):
                logger.info(f"方法 {method} 存在")
            else:
                logger.error(f"方法 {method} 不存在")
                
        return predictor
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None

def test_text_pair_prediction(predictor):
    """测试文本对预测功能"""
    if not predictor:
        logger.error("预测器不可用，跳过文本对预测测试")
        return
    
    # 测试文本
    text1_list = [
        "[room] deluxe queen room",
        "[room] standard twin"
    ]
    
    text2_list = [
        "[room] deluxe room [bed] queen",
        "[room] standard room [bed] twin"
    ]
    
    try:
        # 测试文本对编码
        logger.info("测试文本对编码...")
        logits = predictor.encode_text_pair(text1_list, text2_list)
        logger.info(f"获取到logits: shape={logits.shape}, values={logits.tolist()}")
        
        # 计算相似度
        logger.info("计算相似度...")
        similarities = predictor.calculate_similarity(logits, normalize_to_probability=True)
        logger.info(f"相似度结果: {similarities.tolist()}")
        
        # 应用阈值
        threshold = 0.9
        predictions = (similarities > threshold).int().tolist()
        logger.info(f"阈值 {threshold} 下的预测结果: {predictions}")
        
        return True
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False

def main():
    """主函数"""
    logger.info("开始模型测试...")
    
    # 测试模型加载
    predictor = test_model_loading()
    
    # 测试文本对预测
    if predictor:
        test_text_pair_prediction(predictor)
    
    logger.info("测试完成")

if __name__ == "__main__":
    main() 