# -*- coding: utf-8 -*-
"""
SimCSE模型推理脚本
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# ===================== 配置区域 =====================
# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints/best_model')
#TEST_FILE = os.path.join(BASE_DIR, 'data/all_system_room_match_lable_test.csv')
TEST_FILE = os.path.join(BASE_DIR, 'data/processed_country_test.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'evaluation_results')

# 输出文件配置
OUTPUT_FILES = {   
    'best_results': 'best_results_country.csv',          # 最佳结果文件
    'error_cases': 'error_cases_country.csv',     # 错误案例文件
    # 'best_results': 'best_results.csv',          # 最佳结果文件
    # 'error_cases': 'error_cases.csv',    
    'error_analysis': 'error_analysis_country.json',     # 错误分析结果
    'metrics': 'evaluation_metrics.json'         # 评估指标结果
}

# 模型配置
MAX_LENGTH = 256
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256

# 评估配置
#THRESHOLD_RANGE = np.arange(0.95, 0.96, 0.01)  # 阈值搜索范围
THRESHOLD_RANGE = [0.80]  # 阈值搜索范围

DEFAULT_THRESHOLD = 0.96
METRIC_FOR_BEST = 'precision'  # 'precision' 或 'f1'

# 日志配置
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ===================== 日志设置 =====================
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

class SimCSEPredictor:
    def __init__(self, model_path: str):
        """初始化预测器"""
        self.device = torch.device(DEVICE)
        logger.info(f"Using device: {self.device}")
        
        # 加载模型和分词器
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    def encode_batch(self, texts, batch_size=BATCH_SIZE):
        """批量编码文本"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                batch_embeddings = outputs.last_hidden_state[:, 0]
                embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(embeddings, dim=0)

    def calculate_similarity(self, source_embeddings, target_embeddings):
        """计算余弦相似度"""
        # 对向量进行L2归一化
        source_embeddings = torch.nn.functional.normalize(source_embeddings, p=2, dim=1)
        target_embeddings = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)
        
        # 计算余弦相似度
        similarities = torch.mm(source_embeddings, target_embeddings.t())
        
        similarities = torch.clamp(similarities, min=0.0)   

        return similarities

    def evaluate(self, test_file, threshold=DEFAULT_THRESHOLD, batch_size=BATCH_SIZE):
        """评估模型性能"""
        # 加载测试数据
        test_data = pd.read_csv(test_file)
        logger.info(f"加载测试数据: {len(test_data)} 条")
        
        all_spl_room_type_ids = []
        all_predictions = []
        all_similarities = []
        all_labels = []
        
        # 批量处理数据
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
            batch = test_data.iloc[i:i + batch_size]
            
            try:
                # 编码文本
                source_embeddings = self.encode_batch(batch['spl_room_text'].tolist())
                target_embeddings = self.encode_batch(batch['s_room_text'].tolist())

                # 计算相似度
                similarities = self.calculate_similarity(source_embeddings, target_embeddings)
                pair_similarities = torch.diagonal(similarities)
                
                # 根据阈值进行预测
                predictions = (pair_similarities > threshold).float()
                
                all_spl_room_type_ids.extend(batch['spl_room_type_id'].tolist())
                all_predictions.extend(predictions.numpy())
                all_similarities.extend(pair_similarities.numpy())
                all_labels.extend(batch['label'].tolist())
                
            except Exception as e:
                logger.error(f"处理批次 {i} 时出错: {str(e)}")
                logger.error(f"问题数据: {batch.to_dict()}")
                continue
        
        if not all_predictions:
            logger.error("没有成功处理任何数据")
            return None, None
        
        # 转换为numpy数组
        
        all_spl_room_type_ids = np.array(all_spl_room_type_ids)
        all_predictions = np.array(all_predictions)
        all_similarities = np.array(all_similarities)
        all_labels = np.array(all_labels)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_labels, all_predictions, zero_division=0),
            'similarity_stats': {
                'min': float(np.min(all_similarities)),
                'max': float(np.max(all_similarities)),
                'mean': float(np.mean(all_similarities)),
                'median': float(np.median(all_similarities)),
                'std': float(np.std(all_similarities))
            }
        }
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'spl_room_type_id': test_data['spl_room_type_id'].iloc[:len(all_spl_room_type_ids)],
            'source_text': test_data['spl_room_text'].iloc[:len(all_predictions)],
            'target_text': test_data['s_room_text'].iloc[:len(all_predictions)],
            'similarity': all_similarities,
            'prediction': all_predictions,
            'true_label': all_labels
        })
        
        return metrics, results_df

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化推理器
    inferencer = SimCSEPredictor(MODEL_PATH)
    
    # 尝试不同的阈值
    best_score = 0
    best_threshold = DEFAULT_THRESHOLD
    best_metrics = None
    best_results = None
    
    print("\n正在搜索最佳阈值...")
    for threshold in THRESHOLD_RANGE:
        print(f"\n使用阈值 {threshold:.2f}:")
        metrics, results_df = inferencer.evaluate(TEST_FILE, threshold=threshold)
        
        # 打印评估指标
        print(f"评估指标:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        
        # 根据选定的指标更新最佳结果
        current_score = metrics[METRIC_FOR_BEST]
        if current_score > best_score:
            best_score = current_score
            best_threshold = threshold
            best_metrics = metrics
            best_results = results_df
    
    print(f"\n最佳阈值: {best_threshold:.4f}")
    print("\n最佳结果:")
    print(f"准确率: {best_metrics['accuracy']:.4f}")
    print(f"精确率: {best_metrics['precision']:.4f}")
    print(f"召回率: {best_metrics['recall']:.4f}")
    print(f"F1分数: {best_metrics['f1']:.4f}")
    
    # 保存最佳结果
    results_path = os.path.join(OUTPUT_DIR, OUTPUT_FILES['best_results'])
    best_results.to_csv(results_path, index=False)
    
    # 分析错误案例
    print("\n分析错误案例...")
    error_cases = best_results[best_results['prediction'] != best_results['true_label']]
    error_cases_path = os.path.join(OUTPUT_DIR, OUTPUT_FILES['error_cases'])
    error_cases.to_csv(error_cases_path, index=False)
    
    # 打印错误案例统计
    print(f"\n错误案例数量: {len(error_cases)}")
    print("\n错误类型分布:")
    false_positives = len(error_cases[(error_cases['prediction'] == 1) & (error_cases['true_label'] == 0)])
    false_negatives = len(error_cases[(error_cases['prediction'] == 0) & (error_cases['true_label'] == 1)])
    print(f"假正例 (False Positive): {false_positives} ({false_positives/len(error_cases)*100:.2f}%)")
    print(f"假负例 (False Negative): {false_negatives} ({false_negatives/len(error_cases)*100:.2f}%)")
    
    # 分析相似度分布
    print("\n相似度分布分析:")
    positive_similarities = best_results[best_results['true_label'] == 1]['similarity']
    negative_similarities = best_results[best_results['true_label'] == 0]['similarity']
    
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
    
    # 打印错误案例示例
    print("\n错误案例示例:")
    pd.set_option('display.max_colwidth', None)
    print(error_cases[['source_text', 'target_text', 'similarity', 'prediction', 'true_label']].head().to_string())

    # 保存错误分析结果
    error_analysis = {
        'error_cases': error_cases.to_dict(orient='records'),
        'error_analysis': {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_errors': len(error_cases)
        }
    }
    error_analysis_path = os.path.join(OUTPUT_DIR, OUTPUT_FILES['error_analysis'])
    with open(error_analysis_path, 'w') as f:
        json.dump(error_analysis, f)

    # 保存评估指标结果
    metrics_path = os.path.join(OUTPUT_DIR, OUTPUT_FILES['metrics'])
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main() 