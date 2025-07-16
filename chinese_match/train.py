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
        '--config', type=str, default="/home/maxon/disk2/roomMatch/room_match/chinese_match/config/config_mdeberta.json", help='配置文件路径')
    args = parser.parse_args()

    # 创建训练器，并传递配置覆盖
    trainer = DeBERTaTrainer(args.config)

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
