import numpy as np
import logging
from typing import List, Dict, Optional
import torch
from tqdm import tqdm
import time
from pathlib import Path
import hashlib
import json
from datetime import datetime

# 配置
config = {
    'model_name': 'local-text-encoder',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'cache_dir': '/home/maxon/disk2/roomMatch/room_match/group_room/model_cache',
    'embeddings_cache_dir': '/home/maxon/disk2/roomMatch/room_match/group_room/embeddings_cache',
    'batch_size': 32,
    'cache_max_size': 5000,
    'cache_ttl': 7 * 24 * 60 * 60,
    'embedding_size': 300,
}

# 获取日志记录器
logger = logging.getLogger(__name__)


class SimpleTextEncoder:
    """简单的文本编码器，用于生成文本向量"""

    def __init__(self):
        logger.info(f"初始化简单文本编码器，向量维度: {config['embedding_size']}")
        self.word_vectors = {}

    def _get_word_vector(self, word: str) -> np.ndarray:
        """为单词生成一个向量"""
        if word not in self.word_vectors:
            hash_value = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            np.random.seed(hash_value)
            self.word_vectors[word] = np.random.randn(config['embedding_size'])
            self.word_vectors[word] /= np.linalg.norm(self.word_vectors[word])
        return self.word_vectors[word]

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """将文本编码为向量"""
        result = []
        for text in tqdm(texts):
            words = text.lower().split()
            if not words:
                vector = np.zeros(config['embedding_size'])
            else:
                vector = np.mean([self._get_word_vector(word)
                                 for word in words], axis=0)
                if np.linalg.norm(vector) > 0:
                    vector = vector / np.linalg.norm(vector)
            result.append(vector)
        return np.array(result)


class TextProcessor:
    def __init__(self):
        self._setup_cache_dirs()
        self.model = SimpleTextEncoder()
        self._load_cache()

    def _setup_cache_dirs(self):
        """设置缓存目录"""
        for dir_path in [config['cache_dir'], config['embeddings_cache_dir']]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"缓存目录已设置: {config['cache_dir']}, {config['embeddings_cache_dir']}")

    def _load_cache(self):
        """加载向量缓存"""
        self.cache: Dict[str, Dict] = {}
        cache_file = Path(config['embeddings_cache_dir']) / 'cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                # 清理过期缓存
                current_time = time.time()
                self.cache = {k: v for k, v in self.cache.items()
                              if current_time - v['timestamp'] < config['cache_ttl']}
                logger.info(f"已加载 {len(self.cache)} 条向量缓存")
            except Exception as e:
                logger.warning(f"加载缓存失败: {str(e)}")
                self.cache = {}

    def _save_cache(self):
        """保存向量缓存"""
        try:
            cache_file = Path(config['embeddings_cache_dir']) / 'cache.json'
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存 {len(self.cache)} 条向量缓存")
        except Exception as e:
            logger.error(f"保存缓存失败: {str(e)}")

    def _get_text_hash(self, text: str) -> str:
        """生成文本的哈希值"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取缓存的向量"""
        text_hash = self._get_text_hash(text)
        if text_hash in self.cache:
            cache_data = self.cache[text_hash]
            if time.time() - cache_data['timestamp'] < config['cache_ttl']:
                return np.array(cache_data['embedding'])
        return None

    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """缓存向量"""
        text_hash = self._get_text_hash(text)
        self.cache[text_hash] = {
            'embedding': embedding.tolist(),
            'timestamp': time.time()
        }

        # 如果缓存超过最大大小，删除最旧的条目
        if len(self.cache) > config['cache_max_size']:
            sorted_cache = sorted(self.cache.items(),
                                  key=lambda x: x[1]['timestamp'])
            self.cache = dict(sorted_cache[-config['cache_max_size']:])

    def process_texts(self, texts: List[str]) -> np.ndarray:
        """将房型描述文本转换为向量"""
        try:
            # 预处理文本
            processed_texts = [text.lower().strip() for text in texts]

            # 检查缓存
            embeddings = []
            texts_to_process = []
            texts_indices = []

            for i, text in enumerate(processed_texts):
                cached_embedding = self._get_cached_embedding(text)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                else:
                    texts_to_process.append(text)
                    texts_indices.append(i)

            # 处理未缓存的文本
            if texts_to_process:
                start_time = time.time()
                logger.info(f"处理 {len(texts_to_process)} 条未缓存文本...")

                # 批处理
                batch_embeddings = []
                for i in range(0, len(texts_to_process), config['batch_size']):
                    batch = texts_to_process[i:i + config['batch_size']]
                    batch_embedding = self.model.encode(
                        batch,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    batch_embeddings.append(batch_embedding)

                new_embeddings = np.vstack(batch_embeddings)

                # 缓存新的向量
                for text, embedding in zip(texts_to_process, new_embeddings):
                    self._cache_embedding(text, embedding)

                # 合并结果
                final_embeddings = np.zeros(
                    (len(texts), new_embeddings.shape[1]))
                final_embeddings[texts_indices] = new_embeddings
                cached_indices = list(
                    set(range(len(texts))) - set(texts_indices))
                final_embeddings[cached_indices] = np.array(embeddings)

                end_time = time.time()
                logger.info(f"文本处理完成，用时 {end_time - start_time:.2f} 秒")

                # 定期保存缓存
                self._save_cache()
                return final_embeddings
            else:
                logger.info("所有文本都已在缓存中")
                return np.array(embeddings)

        except Exception as e:
            logger.error(f"文本处理失败: {str(e)}")
            raise


# 创建全局文本处理器实例
text_processor = TextProcessor()
