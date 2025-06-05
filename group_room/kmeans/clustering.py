import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass

# 配置
config = {
    'min_clusters': 2,
    'max_clusters': 10,
    'random_state': 42,
    'n_init': 20,
    'max_iter': 500,
    'init': 'k-means++',
    'min_cluster_size': 2,
    'max_attempts': 3,
}

# 获取日志记录器
logger = logging.getLogger(__name__)


@dataclass
class ClusterMetrics:
    """聚类评估指标"""
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    inertia: float
    cluster_sizes: List[int]
    avg_intra_dist: float
    avg_inter_dist: float


def calculate_cluster_metrics(features: np.ndarray, labels: np.ndarray, kmeans: KMeans) -> ClusterMetrics:
    """计算聚类评估指标"""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters <= 1:
        return ClusterMetrics(
            silhouette=0.0,
            calinski_harabasz=0.0,
            davies_bouldin=float('inf'),
            inertia=kmeans.inertia_,
            cluster_sizes=[len(labels)],
            avg_intra_dist=0.0,
            avg_inter_dist=0.0
        )

    # 计算基本指标
    silhouette = silhouette_score(features, labels)
    calinski = calinski_harabasz_score(features, labels)
    davies = davies_bouldin_score(features, labels)

    # 计算簇大小和距离
    cluster_sizes = [np.sum(labels == i) for i in unique_labels]
    centroids = kmeans.cluster_centers_
    intra_dists = []
    inter_dists = []

    for i in unique_labels:
        cluster_points = features[labels == i]
        intra_dists.append(np.mean(np.linalg.norm(
            cluster_points - centroids[i], axis=1)))

        other_centroids = centroids[unique_labels != i]
        if len(other_centroids) > 0:
            inter_dists.append(np.mean(np.linalg.norm(
                centroids[i] - other_centroids, axis=1)))

    avg_intra_dist = np.mean(intra_dists) if intra_dists else 0.0
    avg_inter_dist = np.mean(inter_dists) if inter_dists else 0.0

    return ClusterMetrics(
        silhouette=silhouette,
        calinski_harabasz=calinski,
        davies_bouldin=davies,
        inertia=kmeans.inertia_,
        cluster_sizes=cluster_sizes,
        avg_intra_dist=avg_intra_dist,
        avg_inter_dist=avg_inter_dist
    )


def find_optimal_clusters(features: np.ndarray, min_cluster_size: int = None) -> Tuple[int, Dict]:
    """使用多个指标找到最佳聚类数"""
    n_samples = features.shape[0]
    min_cluster_size = min_cluster_size or config['min_cluster_size']
    max_clusters = min(config['max_clusters'], n_samples // min_cluster_size)

    if max_clusters < config['min_clusters']:
        return config['min_clusters'], {}

    metrics_dict = {}
    best_score = -float('inf')
    best_n_clusters = config['min_clusters']

    for n_clusters in range(config['min_clusters'], max_clusters + 1):
        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=config['init'],
                random_state=config['random_state'],
                n_init=config['n_init'],
                max_iter=config['max_iter']
            )
            labels = kmeans.fit_predict(features)

            # 计算评估指标
            metrics = calculate_cluster_metrics(features, labels, kmeans)
            metrics_dict[n_clusters] = metrics

            # 综合评分：结合多个指标
            score = (
                metrics.silhouette * 0.4 +
                metrics.calinski_harabasz / 10000 * 0.3 +
                (1 / (metrics.davies_bouldin + 1)) * 0.3
            )

            # 检查是否有太小的簇
            if min(metrics.cluster_sizes) < min_cluster_size:
                continue

            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters

            logger.info(
                f"聚类数 {n_clusters}: 轮廓系数={metrics.silhouette:.4f}, 综合评分={score:.4f}")

        except Exception as e:
            logger.warning(f"评估 {n_clusters} 个聚类时出错: {str(e)}")
            continue

    logger.info(f"最佳聚类数: {best_n_clusters}, 评分: {best_score:.4f}")
    return best_n_clusters, metrics_dict


def calculate_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """计算样本间的余弦相似度矩阵"""
    normalized_features = features / \
        np.linalg.norm(features, axis=1)[:, np.newaxis]
    return normalized_features @ normalized_features.T


def perform_clustering(
    features: np.ndarray,
    n_clusters: int = None,
    min_cluster_size: int = None
) -> Tuple[np.ndarray, int, Dict]:
    """执行KMeans++聚类"""
    try:
        min_cluster_size = min_cluster_size or config['min_cluster_size']

        if n_clusters is None:
            n_clusters, metrics_dict = find_optimal_clusters(
                features, min_cluster_size)
        else:
            metrics_dict = {}

        # 执行聚类
        for attempt in range(config['max_attempts']):
            # 确保随机种子在有效范围内
            random_seed = config['random_state'] + attempt
            if random_seed >= 2**32:
                random_seed = random_seed % (2**32 - 1)

            kmeans = KMeans(
                n_clusters=n_clusters,
                init=config['init'],
                random_state=random_seed,
                n_init=config['n_init'],
                max_iter=config['max_iter']
            )

            labels = kmeans.fit_predict(features)
            metrics = calculate_cluster_metrics(features, labels, kmeans)

            # 检查簇大小
            if min(metrics.cluster_sizes) >= min_cluster_size:
                break

            if attempt < config['max_attempts'] - 1:
                logger.warning(f"第 {attempt + 1} 次尝试产生了过小的簇，重试中...")
                n_clusters -= 1  # 减少聚类数重试

        # 计算相似度
        centroids = kmeans.cluster_centers_
        normalized_centroids = centroids / \
            np.linalg.norm(centroids, axis=1)[:, np.newaxis]
        normalized_features = features / \
            np.linalg.norm(features, axis=1)[:, np.newaxis]

        # 相似度矩阵和样本与质心相似度
        similarity_matrix = calculate_similarity_matrix(features)
        sample_centroid_similarities = normalized_features @ normalized_centroids.T

        # 收集聚类信息
        cluster_info = {
            'metrics': metrics.__dict__,
            'centroids': centroids.tolist(),
            'sample_similarities': similarity_matrix.tolist(),
            'centroid_similarities': sample_centroid_similarities.tolist(),
            'history': metrics_dict
        }

        return labels, n_clusters, cluster_info

    except Exception as e:
        logger.error(f"聚类过程中出错: {str(e)}")
        raise
