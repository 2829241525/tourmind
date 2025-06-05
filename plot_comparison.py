import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# 设置seaborn样式
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
})

# 配置
CONFIG = {
    'input_files': {
        'DeBERTa-v3': 'reports/deberta-v3-base.txt',
        'SimBERT': 'reports/sim_bert.txt',
        'SimRoBERTa': 'reports/sim_roberta.txt'
    },
    'output_dir': 'reports/figures',
    'figure_size': (12, 5),
    'dpi': 300,
    'line_styles': {
        'DeBERTa-v3': ('-', 'o'),      # 实线+圆点
        'SimBERT': ('--', 's'),        # 虚线+方块
        'SimRoBERTa': ('-.', 'D'),     # 点划线+菱形
    },
    'colors': {
        'DeBERTa-v3': '#2166AC',    # 深蓝色
        'SimBERT': '#B2182B',       # 深红色
        'SimRoBERTa': '#238B45'     # 深绿色
    },
    'marker_size': 8,               # 增大标记大小
    'line_width': 1.8,              # 适当减小线宽
    'marker_edge_width': 1.2,       # 减小标记边框宽度
    'marker_interval': 2            # 每隔几个点显示一个标记
}


def extract_metrics(file_path):
    """从文件中提取epoch、loss和accuracy数据"""
    epochs, losses, accuracies = [], [], []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用正则表达式匹配所需数据
        pattern = r'【验证集】评估结果 \(Epoch (\d+)\).*?Loss: ([\d.]+).*?Accuracy: ([\d.]+)'
        matches = re.findall(pattern, content, re.DOTALL)

        for match in matches:
            epochs.append(int(match[0]))
            losses.append(float(match[1]))
            accuracies.append(float(match[2]))

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return [], [], []

    return epochs, losses, accuracies


def create_comparison_plots():
    """创建Loss和Accuracy的对比图"""
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # 创建一个包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=CONFIG['figure_size'])

    # 遍历每个模型的数据
    for model_name, file_path in CONFIG['input_files'].items():
        epochs, losses, accuracies = extract_metrics(file_path)

        if not epochs:
            print(f"警告: 模型 {model_name} 没有数据")
            continue

        # 获取线型配置
        line_style, marker = CONFIG['line_styles'][model_name]
        color = CONFIG['colors'][model_name]

        # 为标记点创建掩码（每隔几个点显示一个标记）
        marker_mask = [i % CONFIG['marker_interval']
                       == 0 for i in range(len(epochs))]

        # 绘制Loss曲线
        ax1.plot(epochs, losses,
                 linestyle=line_style,
                 color=color,
                 label=model_name,
                 linewidth=CONFIG['line_width'])
        # 单独绘制标记点
        ax1.plot([x for i, x in enumerate(epochs) if marker_mask[i]],
                 [x for i, x in enumerate(losses) if marker_mask[i]],
                 marker=marker,
                 linestyle='none',
                 color=color,
                 markersize=CONFIG['marker_size'],
                 markeredgewidth=CONFIG['marker_edge_width'],
                 markerfacecolor=color)

        # 绘制Accuracy曲线
        ax2.plot(epochs, accuracies,
                 linestyle=line_style,
                 color=color,
                 label=model_name,
                 linewidth=CONFIG['line_width'])
        # 单独绘制标记点
        ax2.plot([x for i, x in enumerate(epochs) if marker_mask[i]],
                 [x for i, x in enumerate(accuracies) if marker_mask[i]],
                 marker=marker,
                 linestyle='none',
                 color=color,
                 markersize=CONFIG['marker_size'],
                 markeredgewidth=CONFIG['marker_edge_width'],
                 markerfacecolor=color)

    # 设置Loss子图的属性
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('(a) Loss Comparison')
    ax1.legend(frameon=True, facecolor='white',
               edgecolor='gray', loc='upper right')

    # 设置Accuracy子图的属性
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('(b) Accuracy Comparison')
    ax2.legend(frameon=True, facecolor='white',
               edgecolor='gray', loc='lower right')

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图形
    output_path = os.path.join(CONFIG['output_dir'], 'model_comparison.png')
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    print(f"图形已保存至: {output_path}")

    # 关闭图形
    plt.close()


if __name__ == "__main__":
    create_comparison_plots()
