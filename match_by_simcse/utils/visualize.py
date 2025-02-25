import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import numpy as np

class TrainingVisualizer:
    def __init__(self, log_dir: str):
        """
        初始化可视化器
        Args:
            log_dir: 日志目录路径
        """
        self.log_dir = log_dir
        self.training_data = None
        self.df = None
        self.valid_df = None
        
    def load_training_logs(self, pattern: str = "training_log_*.json"):
        """
        加载所有训练日志并合并
        Args:
            pattern: 日志文件匹配模式
        """
        print(f"Searching for log files in {self.log_dir}")
        log_files = glob.glob(os.path.join(self.log_dir, pattern))
        if not log_files:
            raise FileNotFoundError("No log files found")

        # 按文件创建时间排序
        log_files.sort(key=os.path.getctime)
        print(f"Found {len(log_files)} log files")
        
        # 初始化合并后的数据结构
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
        
        # 读取并合并所有日志文件
        for log_file in log_files:
            print(f"\nProcessing {log_file}:")
            with open(log_file, 'r') as f:
                data = json.load(f)
                print("Data keys:", list(data.keys()))
                for key in merged_data.keys():
                    if key in data and isinstance(data[key], list):
                        print(f"{key}: {len(data[key])} items")
                        merged_data[key].extend(data[key])
        
        print("\nMerged data lengths:")
        for key, value in merged_data.items():
            if isinstance(value, list):
                print(f"{key}: {len(value)} items")
        
        self.training_data = merged_data
        
        # 确保训练相关的数组长度一致
        train_arrays = ['steps', 'epochs', 'train_loss', 'learning_rates', 'time_per_step']
        train_lengths = [len(merged_data[key]) for key in train_arrays]
        if len(set(train_lengths)) > 1:
            min_train_length = min(train_lengths)
            print(f"\n警告：训练数据长度不一致，截断到最短长度: {min_train_length}")
            for key in train_arrays:
                merged_data[key] = merged_data[key][:min_train_length]
        
        # 验证数据可以有不同的长度
        print(f"\n训练步数: {len(merged_data['steps'])}")
        if merged_data['valid_loss']:
            print(f"验证步数: {len(merged_data['valid_loss'])}")

        # 创建DataFrame，只包含长度一致的数据
        df_data = {
            'step': merged_data['steps'],
            'epoch': merged_data['epochs'],
            'train_loss': merged_data['train_loss'],
            'learning_rate': merged_data['learning_rates'],
            'time_per_step': merged_data['time_per_step']
        }
        
        print("\nDataFrame columns lengths:")
        for key, value in df_data.items():
            print(f"{key}: {len(value)} items")
        
        self.df = pd.DataFrame(df_data)
        
        # 使用步骤索引作为时间戳
        self.df['timestamp'] = pd.date_range(
            start=datetime.now() - pd.Timedelta(minutes=len(merged_data['steps'])),
            periods=len(merged_data['steps']),
            freq='1min'
        )
        
        # 按步骤排序
        self.df.sort_values('step', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # 处理验证loss
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
    
    def plot_training_curve(self, save_path: str = None):
        """绘制训练曲线"""
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)
        
        # 1. 绘制训练和验证损失曲线
        ax1.plot(self.training_data['steps'], self.training_data['train_loss'], 'b-', label='Train Loss')
        if self.training_data['valid_loss']:
            valid_steps = np.linspace(
                min(self.training_data['steps']), 
                max(self.training_data['steps']), 
                len(self.training_data['valid_loss'])
            )
            ax1.plot(valid_steps, self.training_data['valid_loss'], 'r-', label='Valid Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 绘制学习率曲线
        ax2.plot(self.training_data['steps'], self.training_data['learning_rates'], 'g-')
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        # 3. 绘制梯度范数
        if self.training_data['grad_norms']:
            ax3.plot(self.training_data['steps'], self.training_data['grad_norms'], 'r-')
            ax3.set_title('Gradient Norm')
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('Norm')
            ax3.grid(True)
        
        # 4. 绘制每步训练时间
        ax4.plot(self.training_data['steps'], self.training_data['time_per_step'], 'm-')
        ax4.set_title('Time per Step')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Seconds')
        ax4.grid(True)
        
        # 调整布局并保存
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"保存训练曲线到: {save_path}")
        plt.close()

    def plot_loss_by_epoch(self, save_path: str = None):
        """按epoch绘制loss变化"""
        plt.figure(figsize=(12, 6))
        
        # 计算每个epoch的平均loss
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

        # 绘制训练loss
        plt.errorbar(mean_steps, mean_losses, yerr=std_losses, fmt='o-', capsize=5,
                    label='Train Loss (mean ± std)')
        
        # 如果有验证loss，也绘制出来
        if self.training_data['valid_loss']:
            valid_steps = np.linspace(min(self.training_data['steps']), 
                                    max(self.training_data['steps']),
                                    len(self.training_data['valid_loss']))
            plt.plot(valid_steps, self.training_data['valid_loss'], 'r*-', 
                    label='Validation Loss')
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss by Epoch')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"保存epoch损失图到: {save_path}")
        plt.close()

    def generate_training_report(self, save_dir: str = None):
        """生成训练报告"""
        if save_dir is None:
            save_dir = os.path.join(self.log_dir, 'reports')
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存图表
        self.plot_training_curve(os.path.join(save_dir, f'training_curve_{timestamp}.png'))
        self.plot_loss_by_epoch(os.path.join(save_dir, f'loss_by_epoch_{timestamp}.png'))
        
        # 计算统计信息
        stats = {
            'total_steps': len(self.training_data['steps']),
            'total_epochs': max(self.training_data['epochs']) + 1,
            'train_loss': {
                'mean': float(np.mean(self.training_data['train_loss'])),
                'std': float(np.std(self.training_data['train_loss'])),
                'min': float(min(self.training_data['train_loss'])),
                'max': float(max(self.training_data['train_loss']))
            },
            'learning_rate': {
                'initial': float(self.training_data['learning_rates'][0]),
                'final': float(self.training_data['learning_rates'][-1])
            }
        }
        
        if self.training_data['valid_loss']:
            stats['valid_loss'] = {
                'mean': float(np.mean(self.training_data['valid_loss'])),
                'std': float(np.std(self.training_data['valid_loss'])),
                'min': float(min(self.training_data['valid_loss'])),
                'max': float(max(self.training_data['valid_loss'])),
                'best': float(min(self.training_data['valid_loss']))
            }
        
        # 保存统计信息
        stats_path = os.path.join(save_dir, f'training_stats_{timestamp}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"保存训练统计信息到: {stats_path}")
            
        return stats

def main():
    """主函数"""
    import argparse

    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(os.path.dirname(current_dir), 'logs')
    reports_dir = os.path.join(log_dir, 'reports')

    parser = argparse.ArgumentParser(description='训练日志可视化工具')
    parser.add_argument('--log_dir', type=str, default=log_dir, help='日志目录路径')
    parser.add_argument('--pattern', type=str, default='training_log_*.json', help='日志文件匹配模式')
    parser.add_argument('--save_dir', type=str, default=reports_dir, help='保存报告的目录')
    args = parser.parse_args()
    
    visualizer = TrainingVisualizer(args.log_dir)
    visualizer.load_training_logs(args.pattern)
    stats = visualizer.generate_training_report(args.save_dir)
    
    # 打印关键统计信息
    print("\n训练统计信息:")
    print(f"总步数: {stats['total_steps']}")
    print(f"总轮数: {stats['total_epochs']}")
    print("\n训练Loss:")
    print(f"  平均值: {stats['train_loss']['mean']:.4f}")
    print(f"  标准差: {stats['train_loss']['std']:.4f}")
    print(f"  最小值: {stats['train_loss']['min']:.4f}")
    print(f"  最大值: {stats['train_loss']['max']:.4f}")
    
    if 'valid_loss' in stats:
        print("\n验证Loss:")
        print(f"  平均值: {stats['valid_loss']['mean']:.4f}")
        print(f"  标准差: {stats['valid_loss']['std']:.4f}")
        print(f"  最佳值: {stats['valid_loss']['best']:.4f}")

if __name__ == '__main__':
    main() 