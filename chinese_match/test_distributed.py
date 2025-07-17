# -*- coding: utf-8 -*-
"""
分布式训练测试脚本
用于测试分布式训练功能是否正常工作
"""

import os
import sys
import logging
import torch
from distributed_trainer import print_distributed_info, get_available_gpus, select_gpus, is_distributed_available

# 设置tokenizer并行性
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_gpu_detection():
    """测试GPU检测功能"""
    print("=" * 50)
    print("测试GPU检测功能")
    print("=" * 50)

    # 打印分布式训练信息
    print_distributed_info()

    # 测试GPU选择
    print("\n测试GPU选择功能:")
    print(f"自动选择: {select_gpus('auto')}")
    print(f"指定GPU 0: {select_gpus('0')}")
    print(f"指定GPU 0,1: {select_gpus('0,1')}")
    print(f"指定GPU范围 0-1: {select_gpus('0-1')}")
    print(f"限制最大GPU数量为1: {select_gpus('auto', max_gpus=1)}")

    # 测试分布式训练可用性
    print(f"\n分布式训练可用: {is_distributed_available()}")


def test_trainer_import():
    """测试训练器导入"""
    print("\n" + "=" * 50)
    print("测试训练器导入")
    print("=" * 50)

    try:
        from trainer import DeBERTaTrainer
        print("✓ DeBERTaTrainer 导入成功")

        # 测试创建训练器（不实际训练）
        print("测试创建训练器...")
        trainer = DeBERTaTrainer(
            config_path="config/config_mdeberta_room_group.json",
            enable_distributed=True,
            gpu_config="auto"
        )
        print("✓ 分布式训练器创建成功")

    except Exception as e:
        print(f"✗ 训练器导入失败: {e}")
        return False

    return True


def test_distributed_modules():
    """测试分布式模块导入"""
    print("\n" + "=" * 50)
    print("测试分布式模块导入")
    print("=" * 50)

    try:
        import torch.distributed as dist
        print("✓ torch.distributed 导入成功")

        import torch.multiprocessing as mp
        print("✓ torch.multiprocessing 导入成功")

        from torch.nn.parallel import DistributedDataParallel as DDP
        print("✓ DistributedDataParallel 导入成功")

        from torch.utils.data import DistributedSampler
        print("✓ DistributedSampler 导入成功")

    except Exception as e:
        print(f"✗ 分布式模块导入失败: {e}")
        return False

    return True


def test_cuda_availability():
    """测试CUDA可用性"""
    print("\n" + "=" * 50)
    print("测试CUDA可用性")
    print("=" * 50)

    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"GPU {i}: {props.name}, 内存: {memory_gb:.1f}GB")

    return torch.cuda.is_available()


def main():
    """主测试函数"""
    print("分布式训练功能测试")
    print("=" * 60)

    # 测试CUDA可用性
    cuda_available = test_cuda_availability()

    # 测试GPU检测
    test_gpu_detection()

    # 测试分布式模块导入
    distributed_ok = test_distributed_modules()

    # 测试训练器导入
    trainer_ok = test_trainer_import()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"CUDA可用: {'✓' if cuda_available else '✗'}")
    print(f"分布式模块: {'✓' if distributed_ok else '✗'}")
    print(f"训练器模块: {'✓' if trainer_ok else '✗'}")

    if cuda_available and distributed_ok and trainer_ok:
        print("\n🎉 所有测试通过！分布式训练功能可以正常使用。")
        print("\n使用方法:")
        print("1. 自动选择GPU: ./run_distributed_train.sh config/config_mdeberta_room_group.json auto")
        print("2. 指定GPU: ./run_distributed_train.sh config/config_mdeberta_room_group.json 0,1")
        print("3. Python脚本: python run_distributed_train.py --gpu-config auto")
    else:
        print("\n❌ 部分测试失败，请检查环境配置。")

    return cuda_available and distributed_ok and trainer_ok


if __name__ == "__main__":
    main()
