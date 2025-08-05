# -*- coding: utf-8 -*-
"""
mDeBERTa模型推理脚本
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import torch.nn.functional as F

# ===================== 配置区域 =====================
# 路径配置
BASE_DIR = "/home/maxon/disk2/roomMatch/room_match/hotel_match"
MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints_hotel_v5/best_model')
# TEST_FILE = os.path.join(BASE_DIR, 'data/processed_country_room_valid.csv')
# TEST_FILE = os.path.join(BASE_DIR, 'data/merged_filtered_result.csv')
TEST_FILE = os.path.join(BASE_DIR, 'data/hotel_samples_test.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'evaluation_results')

# 输出文件配置
OUTPUT_FILES = {
    'best_results': 'best_results.csv',          # 最佳结果文件
    'error_cases': 'error_cases.csv',            # 错误案例文件
    'error_analysis': 'error_analysis.json',     # 错误分析结果
    'metrics': 'evaluation_metrics.json'         # 评估指标结果
}

# 模型配置
MAX_LENGTH = 256
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
USE_FP16 = False

# 评估配置
THRESHOLD_RANGE = [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98]
DEFAULT_THRESHOLD = 0.9
METRIC_FOR_BEST = 'f1'

# 备用配置
TOKENIZER_MODEL_NAME = "microsoft/mdeberta-v3-base"

# 数据字段配置
SOURCE_TEXT_FIELD = 'text1'
TARGET_TEXT_FIELD = 'text2'
ID_FIELD = 'CountryCode'

# 日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MDeBERTaPredictor:
    def __init__(self, model_path: str):
        """初始化预测器"""
        self.device = torch.device(DEVICE)
        logger.info(f"使用设备: {self.device}")

        # 加载配置和模型
        self._load_config(model_path)
        self._load_tokenizer(model_path)
        self._load_model(model_path)

    def _load_config(self, model_path):
        """加载配置文件"""
        config_path = os.path.join(BASE_DIR, 'config', 'config_mdeberta.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.temperature = self.config.get('temperature', 0.05)
        else:
            logger.warning("未找到配置文件，使用默认配置")
            self.temperature = 0.05

    def _load_tokenizer(self, model_path):
        """加载分词器"""
        try:
            logger.info("尝试从本地路径加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False)
            logger.info("成功从本地路径加载分词器")
        except Exception as e:
            logger.warning(f"从本地路径加载分词器失败: {str(e)}")
            logger.info(f"尝试从预训练模型加载分词器: {TOKENIZER_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_MODEL_NAME, use_fast=False)
            logger.info(f"成功从预训练模型加载分词器: {TOKENIZER_MODEL_NAME}")

    def _load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        logger.info("加载mDeBERTa Classifier模型")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path).to(self.device)

        # 处理精度设置
        if USE_FP16:
            self.model = self.model.half()
            logger.info("已将模型转换为半精度(FP16)计算")
        else:
            self.model = self.model.float()
            logger.info("已将模型转换为全精度(float32)计算")

        self.loss_type = 'cross_entropy'

        # 启用梯度检查点以节省显存
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        self.model.eval()

    def encode_text_pair(self, text1_list, text2_list, batch_size=BATCH_SIZE):
        """批量编码文本对"""
        logits_list = []

        for i in range(0, len(text1_list), batch_size):
            batch_text1 = text1_list[i:i + batch_size]
            batch_text2 = text2_list[i:i + batch_size]

            # 编码
            encoded = self.tokenizer(
                batch_text1,
                text_pair=batch_text2,
                padding=True,
                truncation='only_first',
                max_length=MAX_LENGTH,
                return_tensors='pt',
                return_attention_mask=True,
                return_token_type_ids=True,
                verbose=False
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                # 推理
                outputs = self.model(**encoded)

                # 处理异常值
                if torch.isnan(outputs.logits).any():
                    outputs.logits = torch.nan_to_num(
                        outputs.logits, nan=-10.0)
                    logger.warning("检测到NaN值，已替换为安全值")

                # 获取正类的logits得分
                batch_logits = outputs.logits[:, 1]

                # 确保返回的是float32类型
                if batch_logits.dtype != torch.float32:
                    batch_logits = batch_logits.float()

                logits_list.append(batch_logits.cpu())

        return torch.cat(logits_list, dim=0)

    def calculate_similarity(self, source_embeddings, normalize_to_probability=True):
        """计算相似度"""
        # 确保为float32类型并应用sigmoid
        if normalize_to_probability:
            if source_embeddings.dtype != torch.float32:
                source_embeddings = source_embeddings.float()
            return torch.sigmoid(source_embeddings)
        return source_embeddings

    def evaluate(self, test_file, thresholds=None, batch_size=BATCH_SIZE):
        """评估模型在不同阈值下的性能"""
        # 使用默认阈值范围
        thresholds = thresholds or THRESHOLD_RANGE
        logger.info(f"使用阈值范围: {thresholds}")

        # 加载测试数据
        test_data = pd.read_csv(test_file)
        logger.info(f"加载测试数据: {len(test_data)} 条")

        # 初始化结果容器
        all_ids = []
        all_similarities = []
        all_labels = []
        all_predictions = []
        success_count = error_count = 0

        # 批量处理数据
        for i in tqdm(range(0, len(test_data), batch_size), desc="计算文本相似度"):
            batch = test_data.iloc[i:i + batch_size]

            try:
                # 编码和计算相似度
                logits = self.encode_text_pair(
                    batch[SOURCE_TEXT_FIELD].tolist(),
                    batch[TARGET_TEXT_FIELD].tolist()
                )
                similarities = self.calculate_similarity(logits)

                # 默认阈值的预测
                predictions = (similarities > 0.5).int().cpu().numpy()
                all_predictions.extend(predictions)

                # 收集结果
                all_ids.extend(batch[ID_FIELD].tolist())
                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(batch['label'].tolist())
                success_count += len(batch)
            except Exception as e:
                error_count += len(batch)
                logger.error(f"处理批次 {i} 时出错: {str(e)}")
                logger.error(f"问题数据: {batch.head().to_dict()}")
                continue

        logger.info(f"处理完成: 成功 {success_count} 条, 失败 {error_count} 条")

        if not all_similarities:
            logger.error("没有成功处理任何数据")
            return None, None, None

        # 转换为numpy数组
        all_ids = np.array(all_ids)
        all_similarities = np.array(all_similarities)
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        # 计算默认预测的指标
        metrics = self._calculate_metrics(all_labels, all_predictions)
        logger.info("\n基于0.5阈值的预测结果:")
        logger.info(f"准确率: {metrics['accuracy']:.4f}, 精确率: {metrics['precision']:.4f}, " +
                    f"召回率: {metrics['recall']:.4f}, F1分数: {metrics['f1']:.4f}")

        # 为每个阈值计算指标
        threshold_results = {}
        for threshold in thresholds:
            predictions = (all_similarities > threshold).astype(float)
            metrics = self._calculate_metrics(all_labels, predictions)
            metrics['predictions'] = predictions
            threshold_results[threshold] = metrics

            logger.info(f"\n阈值 {threshold:.2f} 的评估结果:")
            logger.info(f"准确率: {metrics['accuracy']:.4f}, 精确率: {metrics['precision']:.4f}, " +
                        f"召回率: {metrics['recall']:.4f}, F1分数: {metrics['f1']:.4f}")

        # 计算相似度统计
        similarity_stats = self._calculate_similarity_stats(
            all_similarities, all_labels)

        # 创建结果DataFrame
        results_df = self._create_results_df(
            test_data, all_ids, all_similarities, all_labels)

        return threshold_results, results_df, similarity_stats

    def _calculate_metrics(self, labels, predictions):
        """计算评估指标"""
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0)
        }

    def _calculate_similarity_stats(self, similarities, labels):
        """计算相似度统计信息"""
        # 基本统计
        stats = {
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'mean': float(np.mean(similarities)),
            'median': float(np.median(similarities)),
            'std': float(np.std(similarities))
        }

        # 按标签分组的统计
        positive_similarities = similarities[labels == 1]
        negative_similarities = similarities[labels == 0]

        logger.info("\n相似度分布分析:")
        logger.info(f"正样本相似度 - 均值: {np.mean(positive_similarities):.4f}, " +
                    f"中位数: {np.median(positive_similarities):.4f}")
        logger.info(f"负样本相似度 - 均值: {np.mean(negative_similarities):.4f}, " +
                    f"中位数: {np.median(negative_similarities):.4f}")

        return stats

    def _create_results_df(self, test_data, ids, similarities, labels):
        """创建结果DataFrame"""
        return pd.DataFrame({
            'id': test_data[ID_FIELD].iloc[:len(ids)],
            'source_text': test_data[SOURCE_TEXT_FIELD].iloc[:len(similarities)],
            'target_text': test_data[TARGET_TEXT_FIELD].iloc[:len(similarities)],
            'similarity': similarities,
            'true_label': labels
        })


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化推理器
    inferencer = MDeBERTaPredictor(MODEL_PATH)

    # 评估模型
    print("\n开始评估模型...")
    threshold_results, results_df, similarity_stats = inferencer.evaluate(
        TEST_FILE)

    if not threshold_results:
        print("评估失败，无法获取结果")
        return

    # 找到最佳阈值
    best_threshold, best_metrics = find_best_threshold(
        threshold_results, METRIC_FOR_BEST)

    # 打印最佳结果
    print(f"\n最佳阈值: {best_threshold:.4f}")
    print("\n最佳结果:")
    print(f"准确率: {best_metrics['accuracy']:.4f}")
    print(f"精确率: {best_metrics['precision']:.4f}")
    print(f"召回率: {best_metrics['recall']:.4f}")
    print(f"F1分数: {best_metrics['f1']:.4f}")

    # 使用最佳阈值的预测结果更新DataFrame
    results_df['prediction'] = best_metrics['predictions']

    # 保存结果和分析错误案例
    save_results(results_df, best_threshold, best_metrics,
                 threshold_results, similarity_stats)
    analyze_errors(results_df)


def find_best_threshold(threshold_results, metric='f1'):
    """找到最佳阈值"""
    best_score = 0
    best_threshold = DEFAULT_THRESHOLD
    best_metrics = None

    print("\n各阈值的评估结果:")
    for threshold, metrics in threshold_results.items():
        print(f"\n阈值 {threshold:.2f}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")

        # 根据选定的指标更新最佳结果
        current_score = metrics[metric]
        if current_score > best_score:
            best_score = current_score
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def save_results(results_df, best_threshold, best_metrics, threshold_results, similarity_stats):
    """保存结果到文件"""
    # 保存最佳结果
    results_path = os.path.join(OUTPUT_DIR, OUTPUT_FILES['best_results'])
    results_df.to_csv(results_path, index=False)

    # 保存错误案例
    error_cases_path = os.path.join(OUTPUT_DIR, OUTPUT_FILES['error_cases'])
    error_cases = results_df[results_df['prediction']
                             != results_df['true_label']]
    error_cases.to_csv(error_cases_path, index=False)

    # 保存评估指标
    metrics_path = os.path.join(OUTPUT_DIR, OUTPUT_FILES['metrics'])

    # 转换为可序列化的格式
    serializable_threshold_results = {}
    for threshold, metrics in threshold_results.items():
        serializable_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                for k, v in metrics.items() if k != 'predictions'}
        serializable_threshold_results[str(threshold)] = serializable_metrics

    serializable_best_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                 for k, v in best_metrics.items() if k != 'predictions'}

    with open(metrics_path, 'w') as f:
        json.dump({
            'threshold_results': serializable_threshold_results,
            'best_threshold': float(best_threshold),
            'best_metrics': serializable_best_metrics,
            'similarity_stats': {k: float(v) if isinstance(v, np.floating) else v
                                 for k, v in similarity_stats.items()}
        }, f, indent=4)


def analyze_errors(results_df):
    """分析错误案例"""
    error_cases = results_df[results_df['prediction']
                             != results_df['true_label']]

    # 错误统计
    print(f"\n错误案例数量: {len(error_cases)}")
    print("\n错误类型分布:")
    false_positives = len(
        error_cases[(error_cases['prediction'] == 1) & (error_cases['true_label'] == 0)])
    false_negatives = len(
        error_cases[(error_cases['prediction'] == 0) & (error_cases['true_label'] == 1)])
    print(
        f"假正例 (False Positive): {false_positives} ({false_positives/len(error_cases)*100:.2f}%)")
    print(
        f"假负例 (False Negative): {false_negatives} ({false_negatives/len(error_cases)*100:.2f}%)")

    # 相似度分析
    analyze_similarity_distribution(results_df)

    # 保存错误分析数据
    save_error_analysis(error_cases, false_positives, false_negatives)

    # 打印错误案例示例
    print("\n错误案例示例:")
    pd.set_option('display.max_colwidth', None)
    print(error_cases[['source_text', 'target_text', 'similarity',
          'prediction', 'true_label']].head().to_string())


def analyze_similarity_distribution(results_df):
    """分析相似度分布"""
    positive_similarities = results_df[results_df['true_label']
                                       == 1]['similarity']
    negative_similarities = results_df[results_df['true_label']
                                       == 0]['similarity']

    print("\n正样本相似度统计:")
    print(f"  数量: {len(positive_similarities)}")
    print(f"  最小值: {positive_similarities.min():.4f}")
    print(f"  最大值: {positive_similarities.max():.4f}")
    print(f"  平均值: {positive_similarities.mean():.4f}")
    print(f"  中位数: {positive_similarities.median():.4f}")

    print("\n负样本相似度统计:")
    print(f"  数量: {len(negative_similarities)}")
    print(f"  最小值: {negative_similarities.min():.4f}")
    print(f"  最大值: {negative_similarities.max():.4f}")
    print(f"  平均值: {negative_similarities.mean():.4f}")
    print(f"  中位数: {negative_similarities.median():.4f}")


def save_error_analysis(error_cases, false_positives, false_negatives):
    """保存错误分析数据"""
    error_analysis = {
        'error_analysis': {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_errors': len(error_cases)
        }
    }
    error_analysis_path = os.path.join(
        OUTPUT_DIR, OUTPUT_FILES['error_analysis'])
    with open(error_analysis_path, 'w') as f:
        json.dump(error_analysis, f)


if __name__ == '__main__':
    main()
