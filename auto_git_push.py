#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动推送Python代码到Git仓库的脚本
每天凌晨12点自动执行
"""

import os
import logging
import schedule
import time
import subprocess
from datetime import datetime
from pathlib import Path
import hashlib
import json

# ==================== 配置区域 ====================
CONFIG = {
    # 项目路径配置
    'PROJECT_PATH': '/home/maxon/disk2/roomMatch/room_match',

    # Git配置
    'GIT_REMOTE': 'github',  # 远程仓库名称，使用GitHub
    'GIT_BRANCH': 'dev_rtm',    # 推送分支，使用已存在的dev_rtm分支

    # 文件扫描配置
    'FILE_PATTERNS': ['*.py', '*.sh'],  # 需要追踪的文件模式，包含Python和Shell脚本
    # 排除的目录
    'EXCLUDE_DIRS': ['.git', '__pycache__', '.vscode', 'cache', 'output', 'results'],

    # 日志配置
    'LOG_FILE': '/home/maxon/disk2/roomMatch/room_match/auto_git_push.log',
    'LOG_LEVEL': logging.INFO,

    # 缓存配置
    'CACHE_FILE': '/home/maxon/disk2/roomMatch/room_match/.git_push_cache.json',

    # 定时配置
    'SCHEDULE_TIME': '00:00',  # 每天执行时间 (24小时制)

    # 提交信息模板
    'COMMIT_MESSAGE_TEMPLATE': 'Auto push Python files - {timestamp}',
}

# ==================== 日志配置 ====================


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=CONFIG['LOG_LEVEL'],
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(CONFIG['LOG_FILE'], encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# ==================== 文件缓存管理 ====================


class FileCache:
    """文件缓存管理器，用于增量更新检测"""

    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache_data = self._load_cache()

    def _load_cache(self):
        """加载缓存数据"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"加载缓存文件失败: {e}")
        return {}

    def _save_cache(self):
        """保存缓存数据"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缓存文件失败: {e}")

    def get_file_hash(self, file_path):
        """获取文件哈希值"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"获取文件哈希失败 {file_path}: {e}")
            return None

    def is_file_changed(self, file_path):
        """检查文件是否有变化"""
        current_hash = self.get_file_hash(file_path)
        if current_hash is None:
            return False

        cached_hash = self.cache_data.get(file_path)
        return current_hash != cached_hash

    def update_file_cache(self, file_path):
        """更新文件缓存"""
        file_hash = self.get_file_hash(file_path)
        if file_hash:
            self.cache_data[file_path] = file_hash

    def save(self):
        """保存缓存"""
        self._save_cache()

# ==================== Git操作类 ====================


class GitPusher:
    """Git推送管理器"""

    def __init__(self, project_path):
        self.project_path = project_path
        self.cache = FileCache(CONFIG['CACHE_FILE'])

    def find_python_files(self):
        """查找所有Python文件"""
        python_files = []
        project_path = Path(self.project_path)

        for pattern in CONFIG['FILE_PATTERNS']:
            for file_path in project_path.rglob(pattern):
                # 检查是否在排除目录中
                if any(exclude_dir in file_path.parts for exclude_dir in CONFIG['EXCLUDE_DIRS']):
                    continue
                python_files.append(str(file_path))

        logger.info(f"找到 {len(python_files)} 个Python文件")
        return python_files

    def check_changes(self):
        """检查是否有文件变化"""
        python_files = self.find_python_files()
        changed_files = []

        for file_path in python_files:
            if self.cache.is_file_changed(file_path):
                changed_files.append(file_path)
                self.cache.update_file_cache(file_path)

        # 检查是否有新文件
        if not changed_files:
            # 检查git status中是否有未追踪的Python文件
            try:
                result = subprocess.run(
                    ['git', 'status', '--porcelain'],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    check=True
                )

                for line in result.stdout.split('\n'):
                    if line.strip() and line.endswith('.py'):
                        file_status = line[:2]
                        file_name = line[3:].strip()
                        if file_status in ['??', ' M', 'M ', 'A ', ' A']:
                            file_full_path = os.path.join(
                                self.project_path, file_name)
                            if file_full_path not in changed_files:
                                changed_files.append(file_full_path)
                                self.cache.update_file_cache(file_full_path)

            except subprocess.CalledProcessError as e:
                logger.error(f"检查git状态失败: {e}")

        logger.info(f"检测到 {len(changed_files)} 个文件有变化")
        return changed_files

    def git_add_commit_push(self, changed_files):
        """执行git添加、提交和推送操作"""
        if not changed_files:
            logger.info("没有文件变化，跳过推送")
            return True

        try:
            # 切换到项目目录
            os.chdir(self.project_path)

            # 添加所有Python文件
            for pattern in CONFIG['FILE_PATTERNS']:
                subprocess.run(['git', 'add', pattern], check=True)

            # 检查是否有待提交的更改
            result = subprocess.run(
                ['git', 'diff', '--cached', '--quiet'],
                capture_output=True
            )

            if result.returncode == 0:
                logger.info("没有待提交的更改")
                return True

            # 提交更改
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            commit_message = CONFIG['COMMIT_MESSAGE_TEMPLATE'].format(
                timestamp=timestamp)

            subprocess.run([
                'git', 'commit', '-m', commit_message
            ], check=True)

            logger.info(f"提交成功: {commit_message}")

            # 推送到远程仓库
            subprocess.run([
                'git', 'push', CONFIG['GIT_REMOTE'], CONFIG['GIT_BRANCH']
            ], check=True)

            logger.info("推送到远程仓库成功")

            # 保存缓存
            self.cache.save()
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Git操作失败: {e}")
            return False
        except Exception as e:
            logger.error(f"推送过程中发生错误: {e}")
            return False

    def run_auto_push(self):
        """执行自动推送流程"""
        logger.info("开始执行自动推送任务")

        try:
            # 检查文件变化
            changed_files = self.check_changes()

            # 执行git操作
            success = self.git_add_commit_push(changed_files)

            if success:
                logger.info("自动推送任务完成")
            else:
                logger.error("自动推送任务失败")

        except Exception as e:
            logger.error(f"自动推送任务异常: {e}")

# ==================== 定时任务管理 ====================


def run_scheduled_push():
    """定时执行推送任务"""
    pusher = GitPusher(CONFIG['PROJECT_PATH'])
    pusher.run_auto_push()


def start_scheduler():
    """启动定时任务"""
    logger.info(f"启动自动推送服务，计划执行时间: 每天 {CONFIG['SCHEDULE_TIME']}")

    # 设置定时任务
    schedule.every().day.at(CONFIG['SCHEDULE_TIME']).do(run_scheduled_push)

    # 启动时执行一次检查
    logger.info("启动时执行一次检查...")
    run_scheduled_push()

    # 保持运行
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止服务")
            break
        except Exception as e:
            logger.error(f"定时任务异常: {e}")
            time.sleep(60)

# ==================== 主函数 ====================


def main():
    """主函数"""
    logger.info("="*50)
    logger.info("自动Git推送服务启动")
    logger.info(f"项目路径: {CONFIG['PROJECT_PATH']}")
    logger.info(f"远程仓库: {CONFIG['GIT_REMOTE']}")
    logger.info(f"目标分支: {CONFIG['GIT_BRANCH']}")
    logger.info(f"执行时间: 每天 {CONFIG['SCHEDULE_TIME']}")
    logger.info("="*50)

    try:
        start_scheduler()
    except Exception as e:
        logger.error(f"服务启动失败: {e}")


if __name__ == "__main__":
    main()
