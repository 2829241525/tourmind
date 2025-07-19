# # -*- coding: utf-8 -*-
# """
# mDeBERTa模型推理脚本
# """

# import os
# import json
# import torch
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import logging
# import torch.nn.functional as F

# # ===================== 配置区域 =====================
# # 路径配置
# BASE_DIR = "/home/maxon/disk2/roomMatch/room_match/chinese_match"
# MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints_group_result_fix/best_model')
# #TEST_FILE = os.path.join("/home/maxon/disk2/roomMatch/room_match/rft/process/training_data_20250526_205100.csv")
# # TEST_FILE = os.path.join(BASE_DIR, 'data/processed_country_room_valid.csv')
# # TEST_FILE = os.path.join(BASE_DIR, 'data/merged_filtered_result.csv')
# TEST_FILE = os.path.join(BASE_DIR, 'data/room_group/room_group_result_test.csv')
# #TEST_FILE = os.path.join(BASE_DIR, 'data/all_cn_data_processed_test.csv')
# OUTPUT_DIR = os.path.join(BASE_DIR, 'evaluation_results')


# def get_model_dataset_identifiers():
#     """从路径中提取模型和数据集标识"""
#     # 从MODEL_PATH提取模型标识
#     model_dir = os.path.basename(os.path.dirname(MODEL_PATH))
#     model_id = model_dir.replace('checkpoints_', '').replace('_result', '')

#     # 从TEST_FILE提取数据集标识
#     test_file_name = os.path.basename(TEST_FILE)
#     dataset_id = test_file_name.replace('.csv', '')

#     return model_id, dataset_id


# # 获取标识符
# MODEL_ID, DATASET_ID = get_model_dataset_identifiers()

# # 输出文件配置


# def get_output_files():
#     """根据模型和数据集标识生成输出文件名"""
#     suffix = f"{MODEL_ID}_{DATASET_ID}"
#     return {
#         'best_results': f'best_results_{suffix}.csv',          # 最佳结果文件
#         'error_cases': f'error_cases_{suffix}.csv',            # 错误案例文件
#         'error_analysis': f'error_analysis_{suffix}.json',     # 错误分析结果
#         'metrics': f'evaluation_metrics_{suffix}.json'         # 评估指标结果
#     }


# OUTPUT_FILES = get_output_files()

# # 模型配置
# MAX_LENGTH = 128
# DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# BATCH_SIZE = 256  # 推理时可以使用更大的批次
# USE_FP16 = False  # 禁用半精度计算

# # 评估配置
# THRESHOLD_RANGE = [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98]  # 概率阈值搜索范围
# DEFAULT_THRESHOLD = 0.9  # 默认概率阈值
# METRIC_FOR_BEST = 'f1'  # 'precision' 或 'f1'

# # 日志配置
# LOG_LEVEL = logging.INFO
# LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# # ===================== 日志设置 =====================
# logging.basicConfig(
#     level=LOG_LEVEL,
#     format=LOG_FORMAT
# )
# logger = logging.getLogger(__name__)


# class MDeBERTaPredictor:
#     def __init__(self, model_path: str):
#         """初始化预测器"""
#         self.device = torch.device(DEVICE)
#         logger.info(f"使用设备: {self.device}")

#         # 加载模型和分词器
#         try:
#             if os.path.exists(model_path):
#                 # 加载配置文件
#                 config_path = os.path.join(
#                     BASE_DIR, 'config', 'config_mdeberta.json')
#                 if os.path.exists(config_path):
#                     with open(config_path, 'r') as f:
#                         self.config = json.load(f)
#                     logger.info(f"成功加载配置文件: {config_path}")
#                     self.temperature = self.config.get('temperature', 0.05)
#                 else:
#                     logger.warning("未找到配置文件，使用默认配置")
#                     self.temperature = 0.05

#                 # 加载分类器模型
#                 logger.info("使用mDeBERTa Classifier模型 (cross entropy loss)")
#                 self.model = AutoModelForSequenceClassification.from_pretrained(
#                     model_path).to(self.device)

#                 # 确保模型使用全精度计算
#                 if USE_FP16 is False:
#                     self.model = self.model.float()
#                     logger.info("已将模型转换为全精度(float32)计算")

#                 self.loss_type = 'cross_entropy'

#                 self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#                 logger.info(f"成功从本地路径加载模型: {model_path}")

#                 # 记录模型的数据类型信息
#                 for name, param in self.model.named_parameters():
#                     if param.dtype == torch.float16:
#                         logger.info(f"检测到半精度参数: {name} (dtype={param.dtype})")
#                         if USE_FP16 is False:
#                             logger.warning(f"警告：尽管设置了使用全精度，但仍检测到半精度参数: {name}")
#                         break
#                 else:
#                     logger.info(
#                         f"模型参数数据类型: {next(self.model.parameters()).dtype}")

#             else:
#                 raise FileNotFoundError(f"模型路径不存在: {model_path}")
#         except Exception as e:
#             logger.error(f"加载模型失败: {str(e)}")
#             raise

#         self.model.eval()

#         # 启用梯度检查点以节省显存
#         if hasattr(self.model, 'gradient_checkpointing_enable'):
#             self.model.gradient_checkpointing_enable()
#             logger.info("已启用梯度检查点以节省显存")

#     def encode_text_pair(self, text1_list, text2_list, batch_size=BATCH_SIZE):
#         """批量编码文本对，用于cross_entropy模式"""
#         logits_list = []

#         for i in range(0, len(text1_list), batch_size):
#             batch_text1 = text1_list[i:i + batch_size]
#             batch_text2 = text2_list[i:i + batch_size]

#             # 对于分类器模型，使用文本对输入
#             encoded = self.tokenizer(
#                 batch_text1,
#                 text_pair=batch_text2,
#                 padding=True,
#                 truncation=True,
#                 max_length=MAX_LENGTH,
#                 return_tensors='pt'
#             )

#             encoded = {k: v.to(self.device) for k, v in encoded.items()}

#             with torch.no_grad():
#                 # 移除autocast，直接使用全精度计算
#                 outputs = self.model(**encoded)

#                 # 检查并处理NaN值
#                 if torch.isnan(outputs.logits).any():
#                     # 将NaN值替换为一个安全的负值（表示低概率）
#                     outputs.logits = torch.nan_to_num(
#                         outputs.logits, nan=-10.0)
#                     logger.warning("检测到NaN值，已替换为安全值")

#                 # 获取正类的logits得分
#                 batch_logits = outputs.logits[:, 1]
#                 # 确保返回的是float32类型
#                 if batch_logits.dtype != torch.float32:
#                     batch_logits = batch_logits.float()
#                 logits_list.append(batch_logits.cpu())

#         return torch.cat(logits_list, dim=0)

#     def calculate_similarity(self, source_embeddings, normalize_to_probability=True):
#         """计算相似度，将logits转换为概率值

#         Args:
#             source_embeddings: 源文本嵌入或logits
#             normalize_to_probability: 是否将结果归一化为0-1之间的概率值
#         """
#         # 获取logits作为相似度分数
#         similarities = source_embeddings

#         # 使用sigmoid函数将logits转换为0-1之间的概率值
#         if normalize_to_probability:
#             # 确保张量为float32类型
#             if similarities.dtype != torch.float32:
#                 similarities = similarities.float()
#             similarities = torch.sigmoid(similarities)

#         return similarities

#     def evaluate(self, test_file, thresholds=None, batch_size=BATCH_SIZE):
#         """评估模型在不同阈值下的性能"""
#         # 设置默认阈值范围
#         if thresholds is None:
#             # 对于概率值，使用标准阈值范围
#             thresholds = THRESHOLD_RANGE
#             logger.info(f"使用默认概率阈值范围: {thresholds}")

#         # 加载测试数据
#         test_data = pd.read_csv(test_file)
#         logger.info(f"加载测试数据: {len(test_data)} 条")

#         all_spl_room_type_ids = []
#         all_similarities = []
#         all_labels = []
#         all_predictions = []  # 新增：保存直接的分类预测结果

#         # 跟踪处理统计
#         error_count = 0
#         success_count = 0

#         # 批量处理数据
#         for i in tqdm(range(0, len(test_data), batch_size), desc="计算文本相似度"):
#             batch = test_data.iloc[i:i + batch_size]

#             try:
#                 # 对文本对进行编码并计算相似度
#                 logits = self.encode_text_pair(
#                     batch['spl_room_text'].tolist(),
#                     batch['s_room_text'].tolist()
#                 )
#                 similarities = self.calculate_similarity(
#                     logits, normalize_to_probability=True)

#                 # 使用阈值0.5进行预测
#                 if isinstance(similarities, torch.Tensor):
#                     predictions = (similarities > 0.5).int().cpu().numpy()
#                     all_predictions.extend(predictions)

#                 all_spl_room_type_ids.extend(
#                     batch['spl_room_type_id'].tolist())
#                 all_similarities.extend(similarities.cpu().numpy())
#                 all_labels.extend(batch['label'].tolist())

#                 success_count += len(batch)

#             except Exception as e:
#                 error_count += len(batch)
#                 logger.error(f"处理批次 {i} 时出错: {str(e)}")
#                 logger.error(f"问题数据: {batch.head().to_dict()}")
#                 # 添加更详细的错误跟踪
#                 import traceback
#                 logger.error(f"详细错误: {traceback.format_exc()}")
#                 continue

#         logger.info(f"处理完成: 成功 {success_count} 条, 失败 {error_count} 条")

#         if not all_similarities:
#             logger.error("没有成功处理任何数据")
#             return None, None, None

#         # 转换为numpy数组
#         all_spl_room_type_ids = np.array(all_spl_room_type_ids)
#         all_similarities = np.array(all_similarities)
#         all_labels = np.array(all_labels)
#         all_predictions = np.array(
#             all_predictions) if all_predictions else np.array([])

#         # 输出直接预测的结果
#         if len(all_predictions) > 0:
#             accuracy = accuracy_score(all_labels, all_predictions)
#             precision = precision_score(
#                 all_labels, all_predictions, zero_division=0)
#             recall = recall_score(all_labels, all_predictions, zero_division=0)
#             f1 = f1_score(all_labels, all_predictions, zero_division=0)

#             logger.info("\n基于argmax的预测结果:")
#             logger.info(f"准确率: {accuracy:.4f}")
#             logger.info(f"精确率: {precision:.4f}")
#             logger.info(f"召回率: {recall:.4f}")
#             logger.info(f"F1分数: {f1:.4f}")

#         # 为每个阈值计算指标
#         threshold_results = {}
#         for threshold in thresholds:
#             predictions = (all_similarities > threshold).astype(float)
#             metrics = {
#                 'accuracy': accuracy_score(all_labels, predictions),
#                 'precision': precision_score(all_labels, predictions, zero_division=0),
#                 'recall': recall_score(all_labels, predictions, zero_division=0),
#                 'f1': f1_score(all_labels, predictions, zero_division=0),
#                 'predictions': predictions
#             }
#             threshold_results[threshold] = metrics

#             # 输出每个阈值的详细结果
#             logger.info(f"\n阈值 {threshold:.2f} 的评估结果:")
#             logger.info(f"准确率: {metrics['accuracy']:.4f}")
#             logger.info(f"精确率: {metrics['precision']:.4f}")
#             logger.info(f"召回率: {metrics['recall']:.4f}")
#             logger.info(f"F1分数: {metrics['f1']:.4f}")

#         # 计算相似度统计信息
#         similarity_stats = {
#             'min': float(np.min(all_similarities)),
#             'max': float(np.max(all_similarities)),
#             'mean': float(np.mean(all_similarities)),
#             'median': float(np.median(all_similarities)),
#             'std': float(np.std(all_similarities))
#         }

#         # 分析正负样本的相似度分布
#         positive_similarities = all_similarities[all_labels == 1]
#         negative_similarities = all_similarities[all_labels == 0]

#         logger.info("\n相似度分布分析:")
#         logger.info(
#             f"正样本相似度 - 均值: {np.mean(positive_similarities):.4f}, 中位数: {np.median(positive_similarities):.4f}")
#         logger.info(
#             f"负样本相似度 - 均值: {np.mean(negative_similarities):.4f}, 中位数: {np.median(negative_similarities):.4f}")

#         # 创建基础结果DataFrame
#         results_df = pd.DataFrame({
#             'spl_room_type_id': test_data['spl_room_type_id'].iloc[:len(all_spl_room_type_ids)],
#             'source_text': test_data['spl_room_text'].iloc[:len(all_similarities)],
#             'target_text': test_data['s_room_text'].iloc[:len(all_similarities)],
#             'similarity': all_similarities,
#             'true_label': all_labels
#         })

#         return threshold_results, results_df, similarity_stats


# def main():
#     # 创建输出目录
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # 初始化推理器
#     inferencer = MDeBERTaPredictor(MODEL_PATH)

#     # 计算所有阈值的结果
#     print("\n开始评估模型...")
#     threshold_results, results_df, similarity_stats = inferencer.evaluate(
#         TEST_FILE)

#     # 找到最佳阈值
#     best_score = 0
#     best_threshold = DEFAULT_THRESHOLD

#     print("\n各阈值的评估结果:")
#     for threshold, metrics in threshold_results.items():
#         print(f"\n阈值 {threshold:.2f}:")
#         print(f"  准确率: {metrics['accuracy']:.4f}")
#         print(f"  精确率: {metrics['precision']:.4f}")
#         print(f"  召回率: {metrics['recall']:.4f}")
#         print(f"  F1分数: {metrics['f1']:.4f}")

#         # 根据选定的指标更新最佳结果
#         current_score = metrics[METRIC_FOR_BEST]
#         if current_score > best_score:
#             best_score = current_score
#             best_threshold = threshold
#             best_metrics = metrics

#     print(f"\n最佳阈值: {best_threshold:.4f}")
#     print("\n最佳结果:")
#     print(f"准确率: {best_metrics['accuracy']:.4f}")
#     print(f"精确率: {best_metrics['precision']:.4f}")
#     print(f"召回率: {best_metrics['recall']:.4f}")
#     print(f"F1分数: {best_metrics['f1']:.4f}")

#     # 使用最佳阈值的预测结果更新DataFrame
#     results_df['prediction'] = best_metrics['predictions']

#     # 保存最佳结果
#     results_path = os.path.join(OUTPUT_DIR, OUTPUT_FILES['best_results'])
#     results_df.to_csv(results_path, index=False)

#     # 分析错误案例
#     print("\n分析错误案例...")
#     error_cases = results_df[results_df['prediction']
#                              != results_df['true_label']]
#     error_cases_path = os.path.join(OUTPUT_DIR, OUTPUT_FILES['error_cases'])
#     error_cases.to_csv(error_cases_path, index=False)

#     # 打印错误案例统计
#     print(f"\n错误案例数量: {len(error_cases)}")
#     print("\n错误类型分布:")
#     false_positives = len(
#         error_cases[(error_cases['prediction'] == 1) & (error_cases['true_label'] == 0)])
#     false_negatives = len(
#         error_cases[(error_cases['prediction'] == 0) & (error_cases['true_label'] == 1)])
#     print(
#         f"假正例 (False Positive): {false_positives} ({false_positives/len(error_cases)*100:.2f}%)")
#     print(
#         f"假负例 (False Negative): {false_negatives} ({false_negatives/len(error_cases)*100:.2f}%)")

#     # 分析相似度分布
#     print("\n相似度分布分析:")
#     positive_similarities = results_df[results_df['true_label']
#                                        == 1]['similarity']
#     negative_similarities = results_df[results_df['true_label']
#                                        == 0]['similarity']

#     print("\n正样本相似度统计:")
#     print(f"  数量: {len(positive_similarities)}")
#     print(f"  最小值: {positive_similarities.min():.4f}")
#     print(f"  最大值: {positive_similarities.max():.4f}")
#     print(f"  平均值: {positive_similarities.mean():.4f}")
#     print(f"  中位数: {positive_similarities.median():.4f}")

#     print("\n负样本相似度统计:")
#     print(f"  数量: {len(negative_similarities)}")
#     print(f"  最小值: {negative_similarities.min():.4f}")
#     print(f"  最大值: {negative_similarities.max():.4f}")
#     print(f"  平均值: {negative_similarities.mean():.4f}")
#     print(f"  中位数: {negative_similarities.median():.4f}")

#     # 打印错误案例示例
#     print("\n错误案例示例:")
#     pd.set_option('display.max_colwidth', None)
#     print(error_cases[['source_text', 'target_text', 'similarity',
#           'prediction', 'true_label']].head().to_string())

#     # 保存错误分析结果
#     error_analysis = {
#         'error_analysis': {
#             'false_positives': false_positives,
#             'false_negatives': false_negatives,
#             'total_errors': len(error_cases)
#         },
#         'similarity_stats': similarity_stats
#     }
#     error_analysis_path = os.path.join(
#         OUTPUT_DIR, OUTPUT_FILES['error_analysis'])
#     with open(error_analysis_path, 'w') as f:
#         json.dump(error_analysis, f)

#     # 保存评估指标结果
#     metrics_path = os.path.join(OUTPUT_DIR, OUTPUT_FILES['metrics'])

#     # 将 NumPy 数组转换为 Python 原生类型
#     serializable_threshold_results = {}
#     for threshold, metrics in threshold_results.items():
#         serializable_metrics = {}
#         for key, value in metrics.items():
#             if key != 'predictions':  # 跳过预测数组，因为它太大了
#                 if isinstance(value, np.floating):
#                     serializable_metrics[key] = float(value)
#                 elif isinstance(value, np.integer):
#                     serializable_metrics[key] = int(value)
#                 else:
#                     serializable_metrics[key] = value
#         serializable_threshold_results[str(threshold)] = serializable_metrics

#     # 确保 best_metrics 中的值也是可序列化的
#     serializable_best_metrics = {}
#     for key, value in best_metrics.items():
#         if key != 'predictions':  # 跳过预测数组
#             if isinstance(value, (np.floating, np.integer)):
#                 serializable_best_metrics[key] = float(value)
#             else:
#                 serializable_best_metrics[key] = value

#     with open(metrics_path, 'w') as f:
#         json.dump({
#             'threshold_results': serializable_threshold_results,
#             'best_threshold': float(best_threshold),
#             'best_metrics': serializable_best_metrics,
#             'similarity_stats': {k: float(v) if isinstance(v, np.floating) else v
#                                  for k, v in similarity_stats.items()}
#         }, f, indent=4)


# if __name__ == '__main__':
#     main()
