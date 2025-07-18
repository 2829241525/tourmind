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
    # config_mdeberta   config_mdeberta_room_group
    parser.add_argument(
        '--config', type=str, default="/home/maxon/disk2/roomMatch/room_match/chinese_match/config/config_mdeberta_room_group.json", help='配置文件路径')
    parser.add_argument(
        '--distributed', action='store_true', help='启用分布式训练')
    parser.add_argument(
        '--gpu-ids', type=str, default=None, help='指定使用的GPU IDs，用逗号分隔，如 "0,1,2"')
    args = parser.parse_args()


    # 准备配置覆盖
    config_override = {}

    # 使用单GPU训练
    logger.info("启动单GPU训练...")

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
