# -*- coding: utf-8 -*-
"""
SimCSE模型推理HTTP服务
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
import uvicorn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import re
from pathlib import Path
import torch.nn.functional as F
import logging

# 配置区域
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 文本处理配置
UNKNOWN_TEXT = "[unknown]"
SUPPLY_PREFIX = "[supply]"
ROOM_PREFIX = "[room]"
BED_PREFIX = "[bed]"

# 模型路径映射
MODEL_PATHS = {
    'robert_new': os.path.join(BASE_DIR, 'checkpoints/best_model'),
    'robert': os.path.join(BASE_DIR, 'checkpoints_robert/best_model'),
    'bert': os.path.join(BASE_DIR, 'checkpoints_bert/best_model'),
    '6_lesslr': os.path.join(BASE_DIR, 'checkpoints_6_lesslr')
}
DEFAULT_MODEL = '6_lesslr'

MAPPING_PATH = os.path.join(BASE_DIR, 'data/mapping.xlsx')
MAX_LENGTH = 256
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
DEFAULT_THRESHOLD = 0.9

# 创建FastAPI应用
app = FastAPI(
    title="房间匹配服务",
    description="基于SimCSE的房间文本匹配服务",
    version="1.0.0"
)

# 请求模型
class MatchRequest(BaseModel):
    spl_room_names: List[str]
    spl_room_bed_names: List[str]
    s_room_names: List[str]
    s_room_bed_names: List[str]
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_MODEL

# 响应模型
class MatchResponse(BaseModel):
    similarities: List[float]
    predictions: List[int]

class SimCSEPredictor:
    def __init__(self, model_path: str):
        """初始化预测器"""
        self.device = torch.device(DEVICE)
        print(f"Using device: {self.device}")
        
        # 加载模型和分词器
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.model_path = model_path
        self.temperature = 0.05  # 与训练时保持一致的温度参数

    def encode_batch(self, texts):
        """编码文本"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0]
        
        return embeddings

    def calculate_similarity(self, source_embeddings, target_embeddings):
        """计算余弦相似度"""
        # 对向量进行L2归一化
        source_embeddings = torch.nn.functional.normalize(source_embeddings, p=2, dim=1)
        target_embeddings = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)
        
        # 计算余弦相似度
        similarities = torch.mm(source_embeddings, target_embeddings.t())
        
        # 将相似度值从[-1, 1]映射到[0, 1]
        similarities = torch.clamp(similarities, min=0.0)
        
        return similarities

    def predict(self, source_texts: List[str], target_texts: List[str], threshold: float = DEFAULT_THRESHOLD):
        """预测文本对的匹配情况"""
        if len(source_texts) != len(target_texts):
            raise ValueError("源文本和目标文本数量必须相同")

        # 编码文本
        source_embeddings = self.encode_batch(source_texts)
        target_embeddings = self.encode_batch(target_texts)

        # 计算相似度
        similarities = self.calculate_similarity(source_embeddings, target_embeddings)
        pair_similarities = torch.diagonal(similarities)

        # 根据阈值进行预测
        predictions = (pair_similarities > threshold).float()

        return {
            'similarities': pair_similarities.cpu().numpy().tolist(),
            'predictions': predictions.cpu().numpy().tolist()
        }
        
    def predict2(self, source_texts: List[str], target_texts: List[str], threshold: float = DEFAULT_THRESHOLD):
        """预测文本对的匹配情况"""
        if len(source_texts) != len(target_texts):
            raise ValueError("源文本和目标文本数量必须相同")

        # 编码文本
        source_embeddings = self.encode_batch(source_texts)
        target_embeddings = self.encode_batch(target_texts)

        # 对向量进行L2归一化
        source_embeddings = F.normalize(source_embeddings, p=2, dim=1)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)

        # 计算相似度（使用与训练时相同的温度参数）
        temperature = 0.1  # 与训练时保持一致
        similarities = torch.mm(source_embeddings, target_embeddings.T) / temperature
        pair_dig = torch.diag(similarities)
        pair_similarities = torch.sigmoid(pair_dig)


        # 根据阈值进行预测
        predictions = (pair_similarities > threshold).float()

        return {
            'similarities': pair_similarities.cpu().numpy().tolist(),
            'predictions': predictions.cpu().numpy().tolist(),
            'threshold': threshold
        }


class DataCleaner:
    def __init__(self, mapping_file: Path):
        self.mapping_file = mapping_file
        self.room_rules, self.bed_rules = self._load_replacement_rules()

    def _load_replacement_rules(self) -> Tuple[dict, List[Tuple[str, str, int]]]:
        # 定义映射替换模式
        xlsx = pd.ExcelFile(self.mapping_file)
        sheet_names = xlsx.sheet_names
        room_rules = {}
        bed_rules = []

        # 按照sheet顺序构建字典
        for sheet_idx, sheet_name in enumerate(sheet_names, 1):
            sheet_data = pd.read_excel(xlsx, sheet_name=sheet_name)
            if not {'key', 'value'}.issubset(sheet_data.columns):
                continue

            # 处理每一行数据
            current_sheet_rules = []
            for _, row in sheet_data.iterrows():
                key = str(row['key']).strip().lower()
                values = [v.strip().strip('"') for v in str(row['value']).split(',')]
                rules = [(value.lower(), key, len(value)) for value in values if value]
                current_sheet_rules.extend(rules)
            
            # 按原文本长度降序排序当前sheet的规则
            current_sheet_rules = sorted(current_sheet_rules, key=lambda x: x[2], reverse=True)
            
            # 如果是bed工作表，单独处理床型
            if sheet_name.lower() == 'bed':
                bed_rules = current_sheet_rules
                room_rules[sheet_name.lower()] = current_sheet_rules
            else:
                # 其他工作表作为房型规则，使用sheet_name作为key
                room_rules[sheet_name.lower()] = current_sheet_rules

        return room_rules, bed_rules

    def clean_room_text(self, text: str) -> str:
        if pd.isna(text):
            return text

        text_lower = str(text).lower()
        # 遍历每个sheet的规则
        for sheet_name, rules in self.room_rules.items():
            # 遍历当前sheet中的所有规则
            for value, key, _ in rules:
                pattern = re.compile(
                    rf'(?:^|(?<=[^a-zA-Z0-9-]))'  # 开始边界：确保前面是非字母数字和横杠
                    rf'{re.escape(value)}'  # 需要匹配的文本
                    rf'(?:$|(?=[^a-zA-Z0-9-]))',  # 结束边界：确保后面是非字母数字和横杠
                    re.IGNORECASE
                )
                if pattern.search(text_lower):  # 如果找到匹配
                    text_lower = pattern.sub(key, text_lower)
                    break  # 找到匹配后就停止当前sheet的搜索
            
            # 如果在当前sheet中找到了匹配，继续处理下一个sheet的规则
            # 这样可以让不同类型的规则（不同sheet）都能应用到文本上

        return text_lower

    def clean_bed_text(self, text: str) -> str:
        if pd.isna(text):
            return text
        if '[unknown]' in text:
            return text

        text_lower = str(text).lower()
        # 只使用bed工作表的规则
        for value, key, _ in self.bed_rules:
            pattern = re.compile(
                rf'(?:^|(?<=[^a-zA-Z0-9-]))'
                rf'{re.escape(value)}'
                rf'(?:$|(?=[^a-zA-Z0-9-]))',
                re.IGNORECASE
            )
            text_lower = pattern.sub(key, text_lower)

        return text_lower

# 全局预测器实例
predictor = None

# 全局清洗器实例
cleaner = None

@app.on_event("startup")
async def startup_event():
    """启动时加载模型和清洗器"""
    global predictor, cleaner
    model_path = MODEL_PATHS.get(DEFAULT_MODEL)
    predictor = SimCSEPredictor(model_path)
    cleaner = DataCleaner(Path(MAPPING_PATH))

def extract_english(text: str) -> str:
    if pd.isna(text):
        return UNKNOWN_TEXT
    # 首先按|分割，并去除分隔符
    parts = [part.strip() for part in text.split('|')]
    parts = [p for p in parts if p]  # 移除空字符串
    
    if not parts:  # 如果没有有效内容
        return UNKNOWN_TEXT
    
    # 检查是否所有部分都只包含中文
    def contains_only_chinese(text):
        text = text.strip()
        if not text:
            return False
        pattern = re.compile(r'^[\u4e00-\u9fff\s]+$')
        return bool(pattern.match(text))
    
    # 检查是否只包含英文和特殊字符
    def contains_only_english(text):
        text = text.strip()
        if not text:
            return False
        # 移除所有特殊字符和空格后，检查是否只包含英文
        cleaned_text = re.sub(r'[^a-zA-Z]', '', text)
        return bool(cleaned_text) and not re.search(r'[\u4e00-\u9fff]', text)
    
    # 获取文本中英文内容的数量
    def get_english_content_length(text):
        # 移除中文字符
        english_only = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+', '', text)
        # 只保留字母和空格
        english_only = re.sub(r'[^a-zA-Z\s]', '', english_only)
        return len(english_only.strip())
    
    # 如果所有部分都只包含中文，返回所有中文内容
    if all(contains_only_chinese(part) for part in parts):
        return ' '.join(parts)
    
    # 如果所有部分都是英文，检查是否内容一致
    if all(contains_only_english(part) for part in parts):
        # 清理并比较内容
        cleaned_parts = [re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z\s]', '', part)).lower().strip() for part in parts]
        if len(set(cleaned_parts)) == 1:  # 所有内容都相同
            return parts[0].strip()
    
    # 处理混合内容，选择英文内容最多的部分
    english_lengths = [(part, get_english_content_length(part)) for part in parts]
    max_english_length = max(length for _, length in english_lengths)
    
    if max_english_length > 0:  # 如果有英文内容
        # 选择英文内容最多的部分
        selected_parts = [part for part, length in english_lengths if length == max_english_length]
        return selected_parts[0].strip()
    
    # 如果没有找到英文内容，但有中文内容，返回所有内容
    has_chinese = any(contains_only_chinese(part) for part in parts)
    if has_chinese:
        return ' '.join(parts)
    
    # 如果既没有英文也没有中文，尝试保留所有非空内容
    if parts:
        return ' '.join(parts)
    
    return UNKNOWN_TEXT

def process_text_with_prefix(prefix: str, room_name: str, bed_name: str, is_supplier: bool = False) -> str:
    # 提取英文
    room = extract_english(room_name)
    bed = extract_english(bed_name)
    prefix_text = prefix if prefix else UNKNOWN_TEXT
    
    # 只对供应商端数据进行清洗
    if is_supplier:
        if room:
            room = cleaner.clean_room_text(room)
        if bed:
            bed = cleaner.clean_room_text(bed)
    
    # 组合成最终格式
    parts = []
    # if prefix:
    #     parts.append(f"{SUPPLY_PREFIX} {prefix_text}" if prefix_text else "")
    if room:
        parts.append(f"{ROOM_PREFIX} {room}")
    if bed:
        parts.append(f"{BED_PREFIX} {bed}")
    #小写
    parts = [part.lower() for part in parts]
    return " ".join(filter(None, parts))

@app.post("/match/", response_model=MatchResponse)
async def match_rooms(request: MatchRequest):
    """
    房间文本匹配接口
    
    - supplier_names: 供应商名称列表
    - spl_room_names: 供应商房型名称列表
    - spl_room_bed_names: 供应商床型名称列表
    - s_room_names: 母房型名称列表
    - s_room_bed_names: 母床型名称列表
    - threshold: 匹配阈值（可选，默认0.5）
    - model: 使用的模型类型（可选，默认'bert'）
    
    返回：
    - similarities: 相似度列表
    - predictions: 预测结果列表（1表示匹配，0表示不匹配）
    - threshold: 使用的阈值
    """
    try:
        # 检查模型类型是否有效
        model_type = request.model.lower()
        model_path = MODEL_PATHS.get(model_type)
        
        if not model_path:
            # 尝试构建可能的路径
            possible_path = os.path.join(BASE_DIR, f'checkpoints_{model_type}/best_model')
            possible_path2 = os.path.join(BASE_DIR, f'checkpoints/checkpoint_{model_type}')
            if os.path.exists(possible_path):
                model_path = possible_path
            elif os.path.exists(possible_path2):
                model_path = possible_path2
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的模型类型: {model_type}。支持的类型有: {list(MODEL_PATHS.keys())}，或确保路径 {possible_path} 存在"
                )
        logging.info(f"使用模型: {model_path}")
            
        # 如果模型类型与当前加载的不同，重新加载模型
        global predictor
        if predictor.model_path != model_path:
            predictor = SimCSEPredictor(model_path)

        # 检查输入列表长度是否一致
        list_lengths = [
            len(request.spl_room_names),
            len(request.spl_room_bed_names),
            len(request.s_room_names),
            len(request.s_room_bed_names)
        ]
        if len(set(list_lengths)) != 1:
            raise HTTPException(
                status_code=400,
                detail="所有输入列表的长度必须相同"
            )

        # 处理文本
        source_texts = [
            process_text_with_prefix(None, spl_room, spl_bed, True)
            for spl_room, spl_bed in zip(
                request.spl_room_names,
                request.spl_room_bed_names
            )
        ]

        target_texts = [
            process_text_with_prefix(None, s_room, s_bed, False)
            for s_room, s_bed in zip(
                request.s_room_names,
                request.s_room_bed_names
            )
        ]

        result = predictor.predict(
            source_texts,
            target_texts,
            request.threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100) 