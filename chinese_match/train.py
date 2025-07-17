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
from distributed_trainer import create_distributed_trainer, print_distributed_info

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
    parser.add_argument('--enable-distributed',
                        action='store_true', help='启用分布式训练')
    parser.add_argument('--gpu-config', type=str,
                        default='auto', help='GPU配置 (auto, 0,1,2, 0-2等)')
    parser.add_argument('--max-gpus', type=int, default=None, help='最大使用GPU数量')
    args = parser.parse_args()

    # 打印分布式训练信息
    print_distributed_info()

    # 创建训练器
    trainer = DeBERTaTrainer(
        args.config,
        enable_distributed=args.enable_distributed,
        gpu_config=args.gpu_config,
        max_gpus=args.max_gpus
    )

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
