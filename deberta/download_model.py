#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import logging
import argparse
import sys
import time
import subprocess

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 检查和安装依赖
def check_and_install_deps():
    try:
        import huggingface_hub
    except ImportError:
        logger.info("正在安装 huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("无法导入 huggingface_hub，请手动安装: pip install huggingface_hub")
        sys.exit(1)
    
    return True


def download_model_git(model_name="microsoft/mdeberta-v3-base",
                      output_dir="/home/maxon/disk2/roomMatch/room_match/deberta/pretrained_models",
                      use_mirror=True):
    """使用git直接克隆模型仓库"""
    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, model_name.split('/')[-1])
    
    # 设置镜像或原站点URL
    base_url = "https://hf-mirror.com" if use_mirror else "https://huggingface.co"
    git_url = f"{base_url}/{model_name}"
    
    try:
        logger.info(f"使用git克隆模型: {git_url} 到 {save_path}")
        # 如果目录已存在，先删除
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        
        # 执行git克隆
        result = subprocess.run(
            ["git", "clone", git_url, save_path],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            logger.info(f"模型克隆成功: {save_path}")
            return save_path
        else:
            logger.error(f"Git克隆失败: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"使用git下载时出错: {str(e)}")
        return None


def download_model(model_name="microsoft/mdeberta-v3-base",
                   output_dir="/home/maxon/disk2/roomMatch/room_match/deberta/pretrained_models",
                   use_mirror=True,
                   force_download=False,
                   max_retries=3):
    """
    下载模型并保存到指定目录
    
    Args:
        model_name: 模型名称或路径
        output_dir: 保存目录
        use_mirror: 是否使用镜像站点
        force_download: 是否强制重新下载
        max_retries: 最大重试次数
    """
    # 导入必要的库
    from huggingface_hub import snapshot_download
    
    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, model_name.split('/')[-1])
    os.makedirs(save_path, exist_ok=True)
    
    # 如果目录已存在且不强制下载，检查是否已下载
    if os.path.exists(os.path.join(save_path, "config.json")) and not force_download:
        logger.info(f"模型已存在于 {save_path}，跳过下载")
        return save_path
    
    # 尝试使用huggingface_hub API下载
    for attempt in range(max_retries):
        try:
            # 设置镜像站点
            if use_mirror:
                logger.info(f"尝试 #{attempt+1}: 使用 hf-mirror.com 作为镜像站点")
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            else:
                logger.info(f"尝试 #{attempt+1}: 使用官方 huggingface.co 站点")
                if "HF_ENDPOINT" in os.environ:
                    del os.environ["HF_ENDPOINT"]
            
            # 下载模型文件
            logger.info(f"开始下载模型 {model_name} 到 {save_path}")
            
            # 使用snapshot_download下载整个模型仓库
            snapshot_download(
                repo_id=model_name,
                local_dir=save_path,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"模型下载完成，保存在: {save_path}")
            return save_path
        
        except Exception as e:
            logger.warning(f"下载尝试 #{attempt+1} 失败: {str(e)}")
            # 最后一次尝试失败后，尝试使用git方法
            if attempt == max_retries - 1:
                logger.info("尝试使用git方法下载...")
                git_result = download_model_git(model_name, output_dir, use_mirror)
                if git_result:
                    return git_result
                logger.error("所有下载方法均失败")
                raise Exception("无法下载模型，请检查网络连接或手动下载")
            
            # 等待一段时间后重试
            wait_time = (attempt + 1) * 2
            logger.info(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
    
    return None

def update_config(config_path, model_path):
    """更新配置文件中的模型路径"""
    import json
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 更新模型路径
        config['pretrained_model'] = model_path
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
            
        logger.info(f"配置文件已更新: {config_path}")
    except Exception as e:
        logger.error(f"更新配置文件失败: {str(e)}")

def main():
    # 检查依赖
    check_and_install_deps()
    
    parser = argparse.ArgumentParser(description="下载Hugging Face模型")
    parser.add_argument("--model", type=str, default="microsoft/mdeberta-v3-base",
                        help="要下载的模型名称")
    parser.add_argument("--output", type=str, 
                        default="/home/maxon/disk2/roomMatch/room_match/deberta/pretrained_models",
                        help="保存模型的目录")
    parser.add_argument("--no-mirror", action="store_false", dest="use_mirror",
                        help="不使用镜像站点")
    parser.add_argument("--use-official", action="store_false", dest="use_mirror",
                        help="使用官方站点")
    parser.add_argument("--force", action="store_true", 
                        help="强制重新下载模型")
    parser.add_argument("--update-config", type=str, default="",
                        help="要更新的配置文件路径")
    parser.add_argument("--git", action="store_true", 
                        help="使用git方式下载")
    
    args = parser.parse_args()
    
    try:
        # 下载模型
        if args.git:
            model_path = download_model_git(
                model_name=args.model,
                output_dir=args.output,
                use_mirror=args.use_mirror
            )
        else:
            model_path = download_model(
                model_name=args.model,
                output_dir=args.output,
                use_mirror=args.use_mirror,
                force_download=args.force
            )
        
        # 如果指定了配置文件，更新配置
        if model_path and args.update_config:
            update_config(args.update_config, model_path)
            
    except Exception as e:
        logger.error(f"下载过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
