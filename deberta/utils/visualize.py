import os
import json
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import numpy as np
import argparse

# 全局配置
CONFIG = {
    "chart_style": "darkgrid",  # 选项: darkgrid, whitegrid, dark, white, ticks
    "default_figure_size": (10, 6),
    "loss_curve_colors": {
        "train": "#1E88E5",     # 蓝色
        "valid": "#D81B60",     # 红色
        "smooth": "#00ACC1"     # 青色
    },
    "font_size": {
        "title": 16,
        "axis_label": 12,
        "tick_label": 10,
        "legend": 10
    },
    "smoothing_factor": 0.8,    # 更高的值 = 更多平滑，范围 0-1
    "save_image_dpi": 300,      # 图像分辨率
    "default_log_patterns": [   # 自动查找日志文件的模式
        "training_log_*.json",
        "*/training_log_*.json",
        "*/logs/training_log_*.json",
        "train_*.log",
        "*/train_*.log",
        "*/logs/train_*.log",
        "*.json",
        "*.log"
    ],
    "default_report_dir": "reports",  # 报告默认保存目录
    "auto_discover_log_dirs": [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs'),
        os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '..', '..', 'logs'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', '..', '..', 'logs')
    ]
}


class TrainingVisualizer:
    def __init__(self, log_file_path: str = None):
        """
        Initialize the visualizer with a log file
        Args:
            log_file_path: Path to specific log file
        """
        # 设置图表样式
        sns.set_style(CONFIG["chart_style"])

        # 初始化数据结构
        self.training_data = {
            'steps': [],
            'epochs': [],
            'train_loss': [],
            'valid_loss': [],
            'learning_rates': [],
            'time_per_step': [],
            'batch_sizes': [],
            'grad_norms': [],
            'timestamp': []
        }

        self.df = None
        self.valid_df = None

        # 如果提供了日志文件，立即处理
        if log_file_path:
            self.process_log_file(log_file_path)

    def process_log_file(self, log_file_path: str):
        """处理单个日志文件并加载数据"""
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"找不到日志文件: {log_file_path}")

        # 检测日志文件类型
        log_type = self.detect_log_file_type(log_file_path)
        print(f"检测到日志类型: {log_type}")

        # 根据日志类型解析文件
        if log_type == 'json':
            # 处理JSON日志
            try:
                with open(log_file_path, 'r') as f:
                    data = json.load(f)
                    for key in self.training_data.keys():
                        if key in data and isinstance(data[key], list):
                            self.training_data[key] = data[key]
                            print(f"{key}: {len(data[key])} 条记录")
            except json.JSONDecodeError:
                raise ValueError(f"文件 {log_file_path} 不是有效的JSON文件")
            except Exception as e:
                raise ValueError(f"处理JSON文件时出错: {str(e)}")

        elif log_type == 'tqdm':
            # 处理tqdm风格日志
            tqdm_data = self.parse_tqdm_log_file(log_file_path)
            if tqdm_data:
                for key in self.training_data.keys():
                    if key in tqdm_data and isinstance(tqdm_data[key], list):
                        self.training_data[key] = tqdm_data[key]
                        print(f"{key}: {len(tqdm_data[key])} 条记录")
            else:
                raise ValueError(f"无法从日志文件中提取训练数据: {log_file_path}")
        else:
            raise ValueError(f"无法识别的日志文件格式: {log_file_path}")

        # 验证是否获取到了必要的训练数据
        if len(self.training_data['steps']) == 0 or len(self.training_data['train_loss']) == 0:
            raise ValueError("未找到有效的训练数据记录")

        # 确保训练相关数组长度一致
        train_arrays = ['steps', 'epochs', 'train_loss',
                        'learning_rates', 'time_per_step']

        # 过滤掉空数组
        train_arrays = [arr for arr in train_arrays if len(
            self.training_data[arr]) > 0]

        if train_arrays:
            train_lengths = [len(self.training_data[key])
                             for key in train_arrays]
            if len(set(train_lengths)) > 1:
                min_train_length = min(train_lengths)
                print(f"\n警告: 训练数据长度不一致，截断为最短长度: {min_train_length}")
                for key in train_arrays:
                    self.training_data[key] = self.training_data[key][:min_train_length]

            # 创建DataFrame
            df_data = {}
            for key in ['steps', 'epochs', 'train_loss', 'learning_rates', 'time_per_step']:
                if len(self.training_data.get(key, [])) > 0:
                    df_data[key.replace('steps', 'step').replace(
                        'epochs', 'epoch')] = self.training_data[key]

            self.df = pd.DataFrame(df_data)

            # 添加时间戳
            if 'step' in df_data:
                self.df['timestamp'] = pd.date_range(
                    start=datetime.now() -
                    pd.Timedelta(minutes=len(self.training_data['steps'])),
                    periods=len(self.training_data['steps']),
                    freq='1min'
                )

                # 按步骤排序
                self.df.sort_values('step', inplace=True)
                self.df.reset_index(drop=True, inplace=True)

            # 处理验证损失
            if self.training_data['valid_loss']:
                valid_steps = np.linspace(
                    min(self.training_data['steps']),
                    max(self.training_data['steps']),
                    len(self.training_data['valid_loss'])
                )
                self.valid_df = pd.DataFrame({
                    'step': valid_steps,
                    'valid_loss': self.training_data['valid_loss']
                })

        else:
            raise ValueError("未找到有效的训练数据数组")

    def parse_tqdm_log_file(self, log_file):
        """
        Parse a tqdm-style log file with progress bar format like:
        Epoch 1/1:  89%|████████▉ | 148018/166369 [6:37:18<47:30,  6.44it/s, loss=0.0001, lr=1.88e-07]

        Also handles mixed logs with other information

        Returns:
            Dict with extracted training data
        """
        print(f"解析tqdm格式日志文件: {log_file}")

        # 正则表达式模式 - 更健壮的模式以处理混合日志
        epoch_pattern = r'Epoch\s+(\d+)/(\d+)'
        step_pattern = r'\|\s*(\d+)/(\d+)'
        loss_pattern = r'loss=([\d\.e\-+]+)'
        lr_pattern = r'lr=([\d\.e\-+]+)'

        # 数据存储
        steps = []
        epochs = []
        losses = []
        learning_rates = []

        valid_lines = 0
        total_lines = 0

        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    total_lines += 1

                    # 只处理看起来像tqdm输出的行
                    if 'Epoch' not in line or '|' not in line:
                        continue

                    # 提取数据
                    epoch_match = re.search(epoch_pattern, line)
                    step_match = re.search(step_pattern, line)
                    loss_match = re.search(loss_pattern, line)

                    # 必须有步数和轮次
                    if not (epoch_match and step_match):
                        continue

                    valid_lines += 1
                    current_epoch = int(epoch_match.group(1)) - 1  # 0-indexed
                    step = int(step_match.group(1))

                    # 损失值可能不在每一行
                    if loss_match:
                        loss = float(loss_match.group(1))
                    else:
                        # 没有损失值则跳过此行
                        continue

                    steps.append(step)
                    epochs.append(current_epoch)
                    losses.append(loss)

                    # 学习率是可选的
                    lr_match = re.search(lr_pattern, line)
                    if lr_match:
                        try:
                            lr = float(lr_match.group(1))
                            learning_rates.append(lr)
                        except ValueError:
                            learning_rates.append(0.0)  # 解析失败时使用默认值
                    else:
                        learning_rates.append(0.0)  # 没有找到学习率时使用默认值

            print(f"处理了 {total_lines} 行，找到 {valid_lines} 条有效训练记录")

            # 如果没有找到任何有效记录，返回None
            if not steps:
                print("未找到有效的训练记录")
                return None

            # 避免由于进度条更新导致的重复记录，只保留每个步骤的最新记录
            data_df = pd.DataFrame({
                'step': steps,
                'epoch': epochs,
                'loss': losses,
                'lr': learning_rates
            })

            # 按步骤分组并保留最后一条记录
            data_df = data_df.drop_duplicates(subset=['step'], keep='last')

            # 按步骤排序
            data_df = data_df.sort_values('step')

            # 打印一些统计信息
            print(
                f"去重后: {len(data_df)} 条记录，步数范围: {data_df['step'].min()}-{data_df['step'].max()}")
            print(
                f"损失范围: {data_df['loss'].min():.6f}-{data_df['loss'].max():.6f}")

            # 转换回列表
            return {
                'steps': data_df['step'].tolist(),
                'epochs': data_df['epoch'].tolist(),
                'train_loss': data_df['loss'].tolist(),
                'learning_rates': data_df['lr'].tolist(),
                'valid_loss': [],  # 没有验证损失
                'time_per_step': [0.0] * len(data_df),  # 占位
                'batch_sizes': [],
                'grad_norms': []
            }

        except Exception as e:
            import traceback
            print(f"解析tqdm日志文件时出错: {str(e)}")
            print(traceback.format_exc())

        return None

    def is_json_file(self, file_path):
        """Check if a file is a valid JSON file"""
        try:
            with open(file_path, 'r') as f:
                # Try to read and parse the first character
                first_char = f.read(1).strip()
                if first_char in ['{', '[']:
                    return True
            return False
        except Exception:
            return False

    def detect_log_file_type(self, file_path):
        """
        Detect the type of log file
        Returns: 'json', 'tqdm', or 'unknown'
        """
        # 检查是否是tqdm格式日志
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = 0
                for line in f:
                    line_count += 1
                    if 'Epoch' in line and '|' in line and ('it/s' in line or 'loss=' in line):
                        print(f"发现tqdm格式进度条在第{line_count}行")
                        return 'tqdm'
                    if line_count > 1000:  # 只检查前1000行
                        break
        except Exception as e:
            print(f"检查tqdm格式时出错: {str(e)}")

        # 检查是否为JSON
        try:
            with open(file_path, 'r') as f:
                first_char = f.read(1).strip()
                if first_char in ['{', '[']:
                    return 'json'
        except Exception:
            pass

        return 'unknown'

    def find_log_files_automatically(self):
        """
        Automatically find log files in common locations
        Returns:
            List of found log files
        """
        found_files = []
        search_dirs = [
            os.getcwd(),  # Current directory
            os.path.join(os.getcwd(), 'logs'),  # logs subdirectory
            os.path.join(os.getcwd(), '..', 'logs'),  # Parent's logs directory
            os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))),  # Project root
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), 'logs')  # Project's logs dir
        ]

        print("Searching for log files automatically...")
        for directory in search_dirs:
            if os.path.exists(directory) and os.path.isdir(directory):
                for pattern in CONFIG["default_log_patterns"]:
                    matching_files = glob.glob(
                        os.path.join(directory, pattern))
                    if matching_files:
                        found_files.extend(matching_files)
                        print(
                            f"Found {len(matching_files)} files in {directory} with pattern {pattern}")

        # Sort by modification time to prefer newer files
        if found_files:
            found_files.sort(key=os.path.getmtime, reverse=True)
            print(
                f"Auto-detected {len(found_files)} log files. Using most recent: {found_files[0]}")
            return found_files

        return []

    def load_training_logs(self, pattern: str = "training_log_*.json"):
        """
        Load and merge all training logs
        Args:
            pattern: Log file matching pattern (used only when log_file is not specified)
        """
        log_files = []

        # Case 1: Single log file specified
        if self.log_file:
            if os.path.isfile(self.log_file):
                log_files = [self.log_file]
                print(f"Using specified log file: {self.log_file}")
            else:
                raise FileNotFoundError(
                    f"Specified log file not found: {self.log_file}")

        # Case 2: Search for log files in directory
        elif self.log_dir:
            if os.path.isdir(self.log_dir):
                print(f"Searching for log files in: {self.log_dir}")
                log_files = glob.glob(os.path.join(self.log_dir, pattern))
                if not log_files:
                    raise FileNotFoundError(
                        f"No log files matching '{pattern}' found in {self.log_dir}")
                print(f"Found {len(log_files)} log files")
            elif os.path.isfile(self.log_dir):
                # Backward compatibility: if log_dir is actually a file
                log_files = [self.log_dir]
                print(
                    f"Warning: --log_dir was given a file path. Using it as a log file: {self.log_dir}")
            else:
                raise FileNotFoundError(
                    f"No such directory or file: {self.log_dir}")
        # Case 3: Auto-detect log files
        else:
            log_files = self.find_log_files_automatically()
            if not log_files:
                raise FileNotFoundError(
                    "No log files found automatically. Please specify --log_dir or --log_file")

        if not log_files:
            raise FileNotFoundError("No log files found")

        # Sort by creation time
        log_files.sort(key=os.path.getctime)

        # Initialize merged data structure
        merged_data = {
            'steps': [],
            'epochs': [],
            'train_loss': [],
            'valid_loss': [],
            'learning_rates': [],
            'time_per_step': [],
            'batch_sizes': [],
            'grad_norms': [],
            'timestamp': []
        }

        # Read and merge all log files
        for log_file in log_files:
            print(f"\nProcessing file {log_file}:")

            log_type = self.detect_log_file_type(log_file)
            print(f"Detected log type: {log_type}")

            if log_type == 'json':
                # Process JSON log file
                try:
                    with open(log_file, 'r') as f:
                        data = json.load(f)
                        print("Data keys: ", list(data.keys()))
                        for key in merged_data.keys():
                            if key in data and isinstance(data[key], list):
                                print(f"{key}: {len(data[key])} records")
                                merged_data[key].extend(data[key])
                except json.JSONDecodeError:
                    print(
                        f"Error: {log_file} is not a valid JSON file. Skipping.")
                    continue
                except Exception as e:
                    print(f"Error processing {log_file}: {str(e)}. Skipping.")
                    continue

            elif log_type == 'tqdm':
                # Process tqdm-style log file
                tqdm_data = self.parse_tqdm_log_file(log_file)
                if tqdm_data:
                    for key in merged_data.keys():
                        if key in tqdm_data and isinstance(tqdm_data[key], list):
                            print(f"{key}: {len(tqdm_data[key])} records")
                            merged_data[key].extend(tqdm_data[key])
                else:
                    print(
                        f"Could not extract training data from {log_file}. Skipping.")
                    continue

            else:
                print(f"Unknown log file format for {log_file}. Skipping.")
                continue

        if all(len(merged_data[key]) == 0 for key in ['steps', 'train_loss']):
            raise ValueError("No valid training data found in log files")

        print("\nMerged data lengths:")
        for key, value in merged_data.items():
            if isinstance(value, list):
                print(f"{key}: {len(value)} records")

        self.training_data = merged_data

        # Ensure training related arrays have consistent lengths
        train_arrays = ['steps', 'epochs', 'train_loss',
                        'learning_rates', 'time_per_step']

        # Filter out empty arrays
        train_arrays = [
            arr for arr in train_arrays if len(merged_data[arr]) > 0]

        if train_arrays:
            train_lengths = [len(merged_data[key]) for key in train_arrays]
            if len(set(train_lengths)) > 1:
                min_train_length = min(train_lengths)
                print(
                    f"\nWarning: Training data lengths are inconsistent, truncating to shortest length: {min_train_length}")
                for key in train_arrays:
                    merged_data[key] = merged_data[key][:min_train_length]

            # Validation data can have different length
            print(f"\nTraining steps: {len(merged_data['steps'])}")
            if merged_data['valid_loss']:
                print(f"Validation steps: {len(merged_data['valid_loss'])}")

            # Create DataFrame with consistent length data
            df_data = {}
            for key in ['step', 'epoch', 'train_loss', 'learning_rate', 'time_per_step']:
                corresponding_key = key if key == 'step' else key.replace(
                    'step', 'steps')
                if len(merged_data.get(corresponding_key, [])) > 0:
                    df_data[key] = merged_data[corresponding_key]

            print("\nDataFrame column lengths:")
            for key, value in df_data.items():
                print(f"{key}: {len(value)} records")

            self.df = pd.DataFrame(df_data)

            # Use step index as timestamp
            if 'step' in df_data:
                self.df['timestamp'] = pd.date_range(
                    start=datetime.now() -
                    pd.Timedelta(minutes=len(merged_data['steps'])),
                    periods=len(merged_data['steps']),
                    freq='1min'
                )

                # Sort by step
                self.df.sort_values('step', inplace=True)
                self.df.reset_index(drop=True, inplace=True)

            # Process validation loss
            if merged_data['valid_loss']:
                valid_steps = np.linspace(
                    min(merged_data['steps']),
                    max(merged_data['steps']),
                    len(merged_data['valid_loss'])
                )
                self.valid_df = pd.DataFrame({
                    'step': valid_steps,
                    'valid_loss': merged_data['valid_loss']
                })
        else:
            raise ValueError("No valid training data arrays found")

    def plot_training_curve(self, save_path: str = None):
        """绘制训练曲线"""
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练指标', fontsize=CONFIG["font_size"]["title"])

        # 设置风格
        sns.set_style(CONFIG["chart_style"])

        # 1. 绘制训练和验证损失曲线
        ax1.plot(self.training_data['steps'],
                 self.training_data['train_loss'],
                 color=CONFIG["loss_curve_colors"]["train"],
                 linewidth=1.5,
                 label='训练损失')
        if self.training_data['valid_loss']:
            valid_steps = np.linspace(
                min(self.training_data['steps']),
                max(self.training_data['steps']),
                len(self.training_data['valid_loss'])
            )
            ax1.plot(
                valid_steps, self.training_data['valid_loss'],
                color=CONFIG["loss_curve_colors"]["valid"],
                linewidth=1.5,
                label='验证损失')
        ax1.set_title(
            '损失曲线', fontsize=CONFIG["font_size"]["axis_label"])
        ax1.set_xlabel('步数', fontsize=CONFIG["font_size"]["axis_label"])
        ax1.set_ylabel('损失值', fontsize=CONFIG["font_size"]["axis_label"])
        ax1.legend(fontsize=CONFIG["font_size"]["legend"])
        ax1.grid(True, linestyle='-', alpha=0.2)

        # 标记最小损失点
        if self.training_data['train_loss']:
            min_idx = np.argmin(self.training_data['train_loss'])
            min_loss = self.training_data['train_loss'][min_idx]
            min_step = self.training_data['steps'][min_idx]
            ax1.scatter([min_step], [min_loss], color='red', s=40, zorder=5)
            ax1.annotate(f'{min_loss:.5f}',
                         (min_step, min_loss),
                         xytext=(10, -20),
                         textcoords='offset points',
                         fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # 2. 绘制学习率曲线
        ax2.plot(self.training_data['steps'],
                 self.training_data['learning_rates'], 'g-')
        ax2.set_title('学习率',
                      fontsize=CONFIG["font_size"]["axis_label"])
        ax2.set_xlabel('步数', fontsize=CONFIG["font_size"]["axis_label"])
        ax2.set_ylabel('学习率',
                       fontsize=CONFIG["font_size"]["axis_label"])
        ax2.set_yscale('log')
        ax2.grid(True, linestyle='-', alpha=0.2)

        # 3. 绘制梯度范数
        if self.training_data['grad_norms']:
            ax3.plot(self.training_data['steps'],
                     self.training_data['grad_norms'], 'r-')
            ax3.set_title('梯度范数',
                          fontsize=CONFIG["font_size"]["axis_label"])
            ax3.set_xlabel('步数', fontsize=CONFIG["font_size"]["axis_label"])
            ax3.set_ylabel('范数值', fontsize=CONFIG["font_size"]["axis_label"])
            ax3.grid(True, linestyle='-', alpha=0.2)

        # 4. 绘制每步耗时
        ax4.plot(self.training_data['steps'],
                 self.training_data['time_per_step'], 'm-')
        ax4.set_title('每步耗时',
                      fontsize=CONFIG["font_size"]["axis_label"])
        ax4.set_xlabel('步数', fontsize=CONFIG["font_size"]["axis_label"])
        ax4.set_ylabel('秒',
                       fontsize=CONFIG["font_size"]["axis_label"])
        ax4.grid(True, linestyle='-', alpha=0.2)

        # 调整布局并保存
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=CONFIG["save_image_dpi"])
            print(f"训练曲线已保存至: {save_path}")
        plt.close()

    def plot_loss_by_epoch(self, save_path: str = None):
        """绘制按轮次的损失变化"""
        plt.figure(figsize=CONFIG["default_figure_size"])
        sns.set_style(CONFIG["chart_style"])

        # 计算每个轮次的平均损失
        epoch_stats = {}
        for step, epoch, loss in zip(self.training_data['steps'],
                                     self.training_data['epochs'],
                                     self.training_data['train_loss']):
            if epoch not in epoch_stats:
                epoch_stats[epoch] = {'losses': [], 'steps': []}
            epoch_stats[epoch]['losses'].append(loss)
            epoch_stats[epoch]['steps'].append(step)

        epochs = sorted(epoch_stats.keys())
        mean_losses = [np.mean(epoch_stats[e]['losses']) for e in epochs]
        std_losses = [np.std(epoch_stats[e]['losses']) for e in epochs]
        mean_steps = [np.mean(epoch_stats[e]['steps']) for e in epochs]

        # 绘制训练损失
        plt.errorbar(mean_steps, mean_losses, yerr=std_losses, fmt='o-', capsize=5,
                     color=CONFIG["loss_curve_colors"]["train"],
                     label='训练损失 (均值 ± 标准差)')

        # 如果存在验证损失，也进行绘制
        if self.training_data['valid_loss']:
            valid_steps = np.linspace(min(self.training_data['steps']),
                                      max(self.training_data['steps']),
                                      len(self.training_data['valid_loss']))
            plt.plot(valid_steps, self.training_data['valid_loss'],
                     color=CONFIG["loss_curve_colors"]["valid"],
                     marker='*', linestyle='-',
                     label='验证损失')

        plt.xlabel('步数', fontsize=CONFIG["font_size"]["axis_label"])
        plt.ylabel('损失值', fontsize=CONFIG["font_size"]["axis_label"])
        plt.title('按轮次的损失变化', fontsize=CONFIG["font_size"]["title"])
        plt.legend(fontsize=CONFIG["font_size"]["legend"])
        plt.grid(True, linestyle='-', alpha=0.2)

        # 标记最小平均损失点
        if mean_losses:
            min_idx = np.argmin(mean_losses)
            min_loss = mean_losses[min_idx]
            min_step = mean_steps[min_idx]
            plt.scatter([min_step], [min_loss], color='red', s=60, zorder=5)
            plt.annotate(f'{min_loss:.5f}',
                         (min_step, min_loss),
                         xytext=(10, -20),
                         textcoords='offset points',
                         fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=CONFIG["save_image_dpi"])
            print(f"轮次损失图已保存至: {save_path}")
        plt.close()

    def plot_loss_trend(self, save_path: str = None, smoothing: float = None):
        """
        绘制简洁的损失趋势图，仅显示损失和步数之间的关系。
        Args:
            save_path: 保存图像的路径
            smoothing: 平滑因子，None时使用默认配置
        """
        if smoothing is None:
            smoothing = CONFIG["smoothing_factor"]

        # 设置图形大小
        plt.figure(figsize=(10, 6))

        # 使用配置中定义的风格
        sns.set_style(CONFIG["chart_style"])

        steps = self.training_data['steps']
        train_loss = self.training_data['train_loss']

        if not steps or not train_loss:
            print("错误: 没有足够的数据来绘制损失趋势")
            return

        # 过滤异常值
        q75 = np.percentile(train_loss, 75)
        std = np.std(train_loss)
        upper_limit = q75 + 3 * std
        valid_indices = [i for i, loss in enumerate(
            train_loss) if loss <= upper_limit]
        filtered_steps = [steps[i] for i in valid_indices]
        filtered_loss = [train_loss[i] for i in valid_indices]

        # 使用过滤后的数据
        steps = filtered_steps
        train_loss = filtered_loss

        # 计算平滑损失
        smooth_train_loss = []
        last = train_loss[0]
        for point in train_loss:
            smoothed_val = last * smoothing + (1 - smoothing) * point
            smooth_train_loss.append(smoothed_val)
            last = smoothed_val

        # 绘制蓝色损失曲线
        plt.plot(steps, smooth_train_loss,
                 color='#1E88E5',
                 linewidth=1.5)

        # 设置Y轴范围
        y_min = max(0, min(train_loss) * 0.9)
        y_max = min(upper_limit, max(train_loss) * 1.1)
        plt.ylim(y_min, y_max)

        # 添加网格线
        plt.grid(True, linestyle='-', alpha=0.2)

        # 简洁的英文标签 (与图片中保持一致)
        plt.xlabel('Steps', fontsize=10)
        plt.ylabel('Loss', fontsize=10)

        # 标记最小损失点
        min_idx = np.argmin(smooth_train_loss)
        min_loss = smooth_train_loss[min_idx]
        min_step = steps[min_idx]
        plt.scatter([min_step], [min_loss], color='red', s=40, zorder=5)

        # 添加最小损失值文本
        plt.annotate(f'{min_loss:.5f}',
                     (min_step, min_loss),
                     xytext=(10, -20),
                     textcoords='offset points',
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # 紧凑布局
        plt.tight_layout()

        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=CONFIG["save_image_dpi"])
            print(f"损失趋势图已保存至: {save_path}")
        plt.close()

        # 返回损失统计信息
        if len(train_loss) > 1:
            first_loss = train_loss[0]
            last_loss = train_loss[-1]
            min_loss = min(train_loss)
            min_loss_idx = train_loss.index(min_loss)
            min_loss_step = steps[min_loss_idx] if min_loss_idx < len(
                steps) else 0
            loss_reduction = ((first_loss - last_loss) /
                              first_loss) * 100 if first_loss != 0 else 0

            return {
                'first_loss': float(first_loss),
                'last_loss': float(last_loss),
                'min_loss': float(min_loss),
                'min_step': int(min_loss_step),
                'reduction_percent': float(loss_reduction)
            }
        return {}

    def generate_statistics(self):
        """生成训练统计信息"""
        try:
            train_loss = self.training_data['train_loss']
            steps = self.training_data['steps']

            if not train_loss or not steps:
                print("警告：没有足够的训练数据来生成统计信息")
                return {"error": "insufficient_data", "message": "没有足够的训练数据来生成统计信息"}

            # 基本统计信息
            stats = {
                'total_steps': len(steps),
                'total_epochs': 0,
                'train_loss': {
                    'mean': float(np.mean(train_loss)),
                    'median': float(np.median(train_loss)),
                    'std': float(np.std(train_loss)),
                    'min': float(min(train_loss)),
                    'max': float(max(train_loss)),
                    'q25': float(np.percentile(train_loss, 25)),
                    'q75': float(np.percentile(train_loss, 75))
                }
            }

            # 添加轮次信息（如果存在）
            if self.training_data['epochs'] and len(self.training_data['epochs']) > 0:
                stats['total_epochs'] = int(
                    max(self.training_data['epochs']) + 1)

            # 添加学习率信息（如果存在）
            if self.training_data['learning_rates'] and len(self.training_data['learning_rates']) > 0:
                stats['learning_rate'] = {
                    'initial': float(self.training_data['learning_rates'][0]),
                    'final': float(self.training_data['learning_rates'][-1]),
                    'min': float(min(self.training_data['learning_rates'])),
                    'max': float(max(self.training_data['learning_rates']))
                }

            # 计算损失下降百分比
            if len(train_loss) > 1:
                first_loss = train_loss[0]
                last_loss = train_loss[-1]
                min_loss = min(train_loss)
                min_loss_idx = train_loss.index(min_loss)
                min_loss_step = steps[min_loss_idx] if min_loss_idx < len(
                    steps) else 0

                # 添加训练趋势信息
                stats['training_progress'] = {
                    'initial_loss': float(first_loss),
                    'final_loss': float(last_loss),
                    'min_loss': float(min_loss),
                    'min_loss_step': int(min_loss_step),
                    'loss_reduction_percent': float(((first_loss - last_loss) / first_loss) * 100 if first_loss != 0 else 0),
                    'min_loss_reduction_percent': float(((first_loss - min_loss) / first_loss) * 100 if first_loss != 0 else 0)
                }

            # 验证损失统计
            if self.training_data['valid_loss'] and len(self.training_data['valid_loss']) > 0:
                valid_loss = self.training_data['valid_loss']
                stats['valid_loss'] = {
                    'mean': float(np.mean(valid_loss)),
                    'std': float(np.std(valid_loss)),
                    'min': float(min(valid_loss)),
                    'max': float(max(valid_loss)),
                    'best': float(min(valid_loss))
                }

            return stats
        except Exception as e:
            import traceback
            print(f"生成统计信息时出错: {str(e)}")
            print(traceback.format_exc())
            return {"error": "statistics_generation_failed", "message": str(e)}

    def generate_training_report(self, save_dir: str = None):
        """生成训练报告"""
        if save_dir is None:
            save_dir = self.get_default_save_dir()

        os.makedirs(save_dir, exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_data = {}

        # 保存图表
        try:
            self.plot_training_curve(os.path.join(
                save_dir, f'training_curve_{timestamp}.png'))
        except Exception as e:
            print(f"无法生成训练曲线图: {str(e)}")
            report_data['training_curve_error'] = str(e)

        try:
            self.plot_loss_by_epoch(os.path.join(
                save_dir, f'loss_by_epoch_{timestamp}.png'))
        except Exception as e:
            print(f"无法生成epoch损失图: {str(e)}")
            report_data['loss_by_epoch_error'] = str(e)

        try:
            loss_data = self.plot_loss_trend(os.path.join(
                save_dir, f'loss_trend_{timestamp}.png'))
            if loss_data:
                report_data['loss_trend'] = loss_data
        except Exception as e:
            print(f"无法生成损失趋势图: {str(e)}")
            report_data['loss_trend_error'] = str(e)
            import traceback
            print(traceback.format_exc())

        # 计算统计信息
        try:
            stats = self.generate_statistics()
            report_data.update(stats)
        except Exception as e:
            print(f"无法生成统计信息: {str(e)}")
            report_data['statistics_error'] = str(e)
            import traceback
            print(traceback.format_exc())

        # 保存报告数据
        report_path = os.path.join(
            save_dir, f'training_report_{timestamp}.json')
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)
            print(f"训练报告已保存到: {report_path}")
        except Exception as e:
            print(f"保存报告时出错: {str(e)}")

        return report_data

    def get_default_save_dir(self):
        """获取报告默认保存路径"""
        # 检查日志文件所在的目录
        logs_dir = os.path.dirname(os.path.abspath(__file__))

        # 尝试在几个常见位置创建reports目录
        possible_dirs = [
            # 当前目录
            os.path.join(os.getcwd(), CONFIG["default_report_dir"]),
            # 日志文件所在目录
            os.path.join(logs_dir, CONFIG["default_report_dir"]),
            # 父目录
            os.path.join(os.path.dirname(logs_dir),
                         CONFIG["default_report_dir"]),
            # 日志目录
            os.path.join(os.path.dirname(logs_dir), 'logs',
                         CONFIG["default_report_dir"])
        ]

        # 使用第一个存在的目录，或者创建第一个
        for save_dir in possible_dirs:
            os.makedirs(save_dir, exist_ok=True)
            if os.path.isdir(save_dir):
                print(f"使用报告保存目录: {save_dir}")
                return save_dir

        # 默认返回当前目录下的reports
        default_dir = os.path.join(os.getcwd(), CONFIG["default_report_dir"])
        os.makedirs(default_dir, exist_ok=True)
        return default_dir


def find_log_files(directory: str = None):
    """
    在目录中查找日志文件
    Args:
        directory: 要搜索的目录
    Returns:
        日志文件路径列表
    """
    found_files = []

    if directory and os.path.isdir(directory):
        # 在指定目录中搜索
        for pattern in CONFIG["default_log_patterns"]:
            matching_files = glob.glob(os.path.join(directory, pattern))
            if matching_files:
                found_files.extend(matching_files)
                print(
                    f"在 {directory} 中找到 {len(matching_files)} 个匹配 {pattern} 的文件")

    # 按修改时间排序
    if found_files:
        found_files.sort(key=os.path.getmtime, reverse=True)

    return found_files


def main():
    """主函数，运行可视化工具"""
    parser = argparse.ArgumentParser(description='训练日志可视化工具')
    parser.add_argument('--log_file', type=str, default=None,
                        help='训练日志文件路径')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='包含训练日志的目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='保存图表的目录')
    parser.add_argument('--only_loss', action='store_true',
                        help='仅绘制损失趋势图')
    parser.add_argument('--smoothing', type=float, default=None,
                        help=f'损失趋势图的平滑因子 (默认: {CONFIG["smoothing_factor"]})')
    parser.add_argument('--full_report', action='store_true',
                        help='生成包含所有图表和统计数据的完整训练报告')

    args = parser.parse_args()

    # 解析日志文件路径
    log_file_path = None
    if args.log_file:
        log_file_path = os.path.abspath(args.log_file)
    elif args.log_dir:
        log_dir = os.path.abspath(args.log_dir)
        log_files = find_log_files(log_dir)
        if log_files:
            # 获取最新的日志文件
            log_file_path = max(log_files, key=os.path.getmtime)
            print(f"找到最新的日志文件: {log_file_path}")
        else:
            print(f"在目录中未找到日志文件: {log_dir}")
            return
    else:
        # 自动从常见位置发现日志
        log_files = []
        for dir_path in CONFIG["auto_discover_log_dirs"]:
            log_files.extend(find_log_files(dir_path))

        if log_files:
            log_file_path = max(log_files, key=os.path.getmtime)
            print(f"自动发现最新日志文件: {log_file_path}")
        else:
            print("未找到日志文件。请通过--log_file或--log_dir参数指定。")
            return

    # 初始化可视化器
    try:
        visualizer = TrainingVisualizer(log_file_path)
        print(f"成功读取日志文件: {log_file_path}")
        print(f"解析了 {len(visualizer.training_data['steps'])} 条训练记录")
    except Exception as e:
        print(f"无法初始化可视化器: {str(e)}")
        return

    # 设置输出目录
    output_dir = args.output_dir if args.output_dir else visualizer.get_default_save_dir()
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 为文件名生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 生成请求的可视化图
    if args.full_report:
        print("生成完整训练报告...")
        stats = visualizer.generate_training_report(output_dir)
        print(f"训练报告生成完成，包含所有图表和统计信息")

        # 打印关键统计数据概述
        if 'training_progress' in stats:
            progress = stats['training_progress']
            print("\n训练进度概述:")
            print(f"  初始损失: {progress['initial_loss']:.4f}")
            print(f"  最终损失: {progress['final_loss']:.4f}")
            print(f"  最低损失: {progress['min_loss']:.4f}")
            print(f"  损失下降: {progress['loss_reduction_percent']:.2f}%")
            print(f"  最佳下降: {progress['min_loss_reduction_percent']:.2f}%")

    elif args.only_loss:
        print("只生成损失趋势图...")
        output_path = os.path.join(output_dir, f'loss_trend_{timestamp}.png')
        visualizer.plot_loss_trend(output_path, smoothing=args.smoothing)
    else:
        print("生成所有图表...")
        visualizer.plot_training_curve(os.path.join(
            output_dir, f'training_curve_{timestamp}.png'))
        visualizer.plot_loss_by_epoch(os.path.join(
            output_dir, f'loss_by_epoch_{timestamp}.png'))
        visualizer.plot_loss_trend(os.path.join(
            output_dir, f'loss_trend_{timestamp}.png'), smoothing=args.smoothing)

    print(f"所有图表已保存到: {output_dir}")


if __name__ == '__main__':
    main()
