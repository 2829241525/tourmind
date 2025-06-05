#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from deberta.utils.visualize import TrainingVisualizer
import numpy as np
import re


def extract_epoch_data_from_log(log_file):
    """
    Directly extract epoch and loss data from log file to overcome limitations of visualizer
    Returns: Dictionary with epoch data
    """
    epochs_data = {}
    current_epoch = None
    step_pattern = r'Epoch (\d+)/2:\s+\d+%\|.*\|\s+(\d+)/\d+.*loss=([0-9.e-]+), lr=([0-9.e-]+)'

    print(f"Directly parsing log file to extract epochs data: {log_file}")

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(step_pattern, line)
            if match:
                epoch = int(match.group(1)) - 1  # Convert to 0-indexed
                step = int(match.group(2))

                try:
                    loss = float(match.group(3))
                except ValueError:
                    print(f"警告: 无法解析损失值: {match.group(3)}，将跳过此行")
                    continue

                try:
                    # 处理不规范的学习率格式
                    lr_str = match.group(4)
                    if lr_str.endswith('e') and not re.search(r'e[+-]?\d+$', lr_str):
                        # 如果学习率以e结尾但不是有效的科学记数法，添加0
                        lr_str = lr_str + '+00'
                    lr = float(lr_str)
                except ValueError:
                    print(f"警告: 无法解析学习率: {match.group(4)}，将使用0")
                    lr = 0.0

                if epoch not in epochs_data:
                    epochs_data[epoch] = {'steps': [], 'losses': [], 'lr': []}

                # Only add each step once (avoid duplicates from the log)
                if step not in epochs_data[epoch]['steps']:
                    epochs_data[epoch]['steps'].append(step)
                    epochs_data[epoch]['losses'].append(loss)
                    epochs_data[epoch]['lr'].append(lr)

    # Report results
    for epoch, data in epochs_data.items():
        print(f"Extracted Epoch {epoch+1}: {len(data['steps'])} steps")

        # Sort by step
        combined = list(zip(data['steps'], data['losses'], data['lr']))
        combined.sort()  # Sort by step (first element)

        if combined:
            # Unzip the sorted data back into separate lists
            data['steps'], data['losses'], data['lr'] = zip(*combined)

            # Print first few steps as sanity check
            print(f"  First 5 steps: {data['steps'][:5]}")
            print(f"  First 5 losses: {data['losses'][:5]}")

    return epochs_data


def generate_loss_lr_chart(log_file, output_dir=None):
    """
    Only generate loss and learning rate charts

    Args:
        log_file: Path to log file
        output_dir: Output directory
    """
    # Set output directory
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), 'reports')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Set matplotlib style for better aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create the main figure with two plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Extract data directly from log file to get both epochs
    epochs_data = extract_epoch_data_from_log(log_file)

    if not epochs_data:
        print("ERROR: No valid epoch data found. Exiting.")
        return

    # Set colors for different epochs
    epoch_colors = ['#1E88E5', '#D81B60', '#8BC34A', '#FF9800']

    # 1. Plot loss curves - Create a separate plot for each epoch
    max_loss = 0
    min_loss = float('inf')

    # Calculate steps per epoch - assume each epoch has the same number of steps
    steps_per_epoch = max([max(data['steps'])
                          for data in epochs_data.values()], default=169560)
    print(f"Steps per epoch: {steps_per_epoch}")

    for i, (epoch, data) in enumerate(sorted(epochs_data.items())):
        # Plot loss data
        color = epoch_colors[i % len(epoch_colors)]
        ax1.plot(data['steps'], data['losses'],
                 color=color,
                 linewidth=1.5,
                 label=f'Epoch {epoch+1}')

        # Find min non-zero loss
        non_zero_losses = [(idx, step, val) for idx, (step, val) in enumerate(
            zip(data['steps'], data['losses'])) if val > 0.0001]
        if non_zero_losses:
            _, min_step, min_val = min(non_zero_losses, key=lambda x: x[2])
            min_loss_val = min_val
            min_loss = min(min_loss, min_loss_val)
            max_loss = max(max_loss, max(data['losses']))

            # Plot marker at minimum loss
            ax1.scatter([min_step], [min_loss_val], color=color,
                        s=100, zorder=5, marker='*')

            # Add annotation for minimum loss
            ax1.annotate(f'E{epoch+1}: {min_loss_val:.5f}',
                         (min_step, min_loss_val),
                         xytext=(10, 10 if i % 2 == 0 else -30),
                         textcoords='offset points',
                         fontsize=10,
                         weight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            print(
                f"Epoch {epoch+1} minimum loss: {min_loss_val:.5f} at step {min_step}")

    # Calculate loss statistics
    for epoch, data in epochs_data.items():
        # Filter out zeros for average calculation
        filtered = [l for l in data['losses'] if l > 0.0001]
        if filtered:
            mean_loss = np.mean(filtered)
            print(f"Epoch {epoch+1} average loss: {mean_loss:.4f}")

    # Adjust plot settings for loss curve
    y_margin = (max_loss - min_loss) * 0.1  # 10% margin
    ax1.set_ylim([max(0, min_loss - y_margin), max_loss + y_margin])
    ax1.set_xlim([0, steps_per_epoch])

    # Improve X-axis ticks for better readability
    x_ticks = np.linspace(0, steps_per_epoch, 7)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{int(x):,}' for x in x_ticks])

    ax1.set_title('Loss Curve', fontsize=16)
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, linestyle='-', alpha=0.2)

    # 2. Plot learning rate curves - for each epoch separately
    for i, (epoch, data) in enumerate(sorted(epochs_data.items())):
        color = epoch_colors[i % len(epoch_colors)]
        ax2.plot(data['steps'], data['lr'],
                 color=color,
                 linewidth=1.5,
                 label=f'Epoch {epoch+1}')

    # Improve X-axis ticks for learning rate plot
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'{int(x):,}' for x in x_ticks])
    ax2.set_xlim([0, steps_per_epoch])

    ax2.set_title('Learning Rate', fontsize=16)
    ax2.set_xlabel('Steps', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='-', alpha=0.2)
    ax2.legend(fontsize=10, loc='upper right')

    # Adjust tight_layout parameters to ensure there's enough space for annotations
    plt.tight_layout(pad=3.0)

    # Save chart
    output_path = os.path.join(output_dir, f'loss_lr_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loss and learning rate chart saved to: {output_path}")
    plt.close()

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate loss and learning rate charts')
    parser.add_argument('--log_file', type=str, required=True,
                        help='Path to training log file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save charts')

    args = parser.parse_args()

    generate_loss_lr_chart(args.log_file, args.output_dir)
