# -*- coding: utf-8 -*-
"""
DeBERTa模型训练脚本
使用统一的训练器模块训练DeBERTa模型
"""

import os
import argparse
import logging
import torch
import json
from trainer import DeBERTaTrainer, print_gpu_memory, clear_gpu_memory

# 默认配置
DEFAULT_CONFIG = {
    "learning_rate": 1e-5,  # 降低默认学习率
    "max_grad_norm": 0.5,   # 降低梯度裁剪阈值
    "fp16": False,          # 默认禁用混合精度训练
    "enable_gradient_checkpointing": False,  # 默认禁用梯度检查点
    "reduce_learning_rate_for_nan": True  # 启用NaN问题时降低学习率
}

# 设置tokenizer并行性
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(
            os.path.dirname(__file__), 'logs', 'train.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练DeBERTa模型')
    parser.add_argument(
        '--config', type=str, default="/home/maxon/disk2/roomMatch/room_match/rank_room/config/config_mdeberta.json", help='配置文件路径')
    parser.add_argument(
        '--reduce_lr', action='store_false', dest='no_reduce_lr', help='不降低学习率')
    parser.add_argument(
        '--enable_gradient_checkpointing', action='store_true', help='启用梯度检查点')
    parser.add_argument(
        '--enable_fp16', action='store_true', help='启用混合精度训练')
    args = parser.parse_args()

    # 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 准备配置覆盖
    config_override = {}

    # 默认降低学习率，除非指定--reduce_lr参数
    if not args.no_reduce_lr:
        config_override['learning_rate'] = config.get(
            'learning_rate', 2e-5) * 0.1
        config_override['max_grad_norm'] = 0.5
        logger.info(f"使用降低的学习率: {config_override['learning_rate']}")
    else:
        logger.info("使用原始学习率")

    # 默认禁用梯度检查点，除非指定--enable_gradient_checkpointing
    config_override['enable_gradient_checkpointing'] = args.enable_gradient_checkpointing
    if not args.enable_gradient_checkpointing:
        logger.info("禁用梯度检查点")
    else:
        logger.info("启用梯度检查点")

    # 默认禁用混合精度训练，除非指定--enable_fp16
    config_override['fp16'] = args.enable_fp16
    if not args.enable_fp16:
        logger.info("禁用混合精度训练")
    else:
        logger.info("启用混合精度训练")

    # 创建训练器，并传递配置覆盖
    trainer = DeBERTaTrainer(args.config, config_override)

    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.exception(f"训练过程中发生错误: {str(e)}")
    finally:
        # 清理GPU内存
        clear_gpu_memory()
        print_gpu_memory()


if __name__ == "__main__":
    main()
