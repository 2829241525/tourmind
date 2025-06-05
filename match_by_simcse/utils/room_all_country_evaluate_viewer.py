"""
房型匹配分析工具 - 支持单酒店分析和批量国家分析
"""

import streamlit as st
# 设置页面配置
st.set_page_config(
    page_title="房型匹配分析工具",
    layout="wide",
    page_icon="🏨"
)

import requests
import json
import pandas as pd
from typing import Dict, List, Optional, Any
import warnings
from dataclasses import dataclass
from urllib3.exceptions import InsecureRequestWarning
import os
import logging
import sys
import debugpy
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# 使用绝对导入
from match_by_simcse.inference import SimCSEPredictor
from match_by_simcse.api_server import process_text_with_prefix, extract_english, DataCleaner

# 添加基础路径常量/home/maxon/disk2/roomMatch/room_match/match_by_simcse/checkpoints_6_lesslr
BASE_DIR = "/home/maxon/disk2/roomMatch/room_match/"
MODEL_PATH = os.path.join(BASE_DIR, "match_by_simcse/checkpoints_512batch_5e/best_model")
# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('room_type_viewer.log')
    ]
)
logger = logging.getLogger(__name__)

# 设置 urllib3 的日志级别为 WARNING，这样就不会显示 DEBUG 级别的连接日志
logging.getLogger("urllib3").setLevel(logging.WARNING)

# 忽略SSL警告
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# 启用调试
def setup_debugger():
    """设置调试器"""
    try:
        if not debugpy.is_client_connected():
            debugpy.listen(("localhost", 5679))
            logger.info("调试器已启动，等待连接...")
    except Exception as e:
        logger.error(f"启动调试器时发生错误: {str(e)}")

# 数据类型定义
@dataclass
class RoomType:
    """房型数据结构"""
    roomTypeName: str
    roomTypeNameEn: str = ""
    bedType: str = ""
    bedTypeEn: str = ""
    roomTypeListType: str = ""
    status: int = 0

@dataclass
class MatchResult:
    """匹配结果数据结构"""
    spl_room_name: str
    spl_bed_type: str
    spl_room_name_en: str
    spl_bed_type_en: str
    standard_room_name: str
    standard_bed_type: str
    match_score: float

def load_supplier_mapping() -> Dict[str, List[str]]:
    """
    加载供应商映射关系
    从supplier_type.xlsx文件中读取供应商ID和名称的映射关系
    """
    try:
        # 读取Excel文件
        file_path = os.path.join(BASE_DIR, "match_by_simcse/data/supplier_type.xlsx")
        
        if not os.path.exists(file_path):
            st.warning(f"供应商映射文件不存在: {file_path}")
            return {}
            
        df = pd.read_excel(file_path)  # 使用 read_excel 替代 read_csv
        
        # 创建映射字典
        mapping = {}
        for _, row in df.iterrows():
            supplier_id = str(row['supplier_id'])
            supplier_name = row['name']
            mapping[supplier_id] = [supplier_name]
        
        # 添加默认值
        mapping["0"] = list(df['name'].unique())
        
        return mapping
    except Exception as e:
        st.error(f"加载供应商映射文件失败: {str(e)}")
        logger.error(f"加载供应商映射文件失败: {str(e)}", exc_info=True)  # 添加详细日志
        return {}

# 加载供应商映射
SUPPLIER_MAPPING = load_supplier_mapping()

def load_hotel_data() -> pd.DataFrame:
    """
    加载酒店数据
    从匹配测试酒店.csv文件中读取酒店ID、名称和国家代码
    """
    try:
        # 读取CSV文件
        file_path = os.path.join(BASE_DIR, "match_by_simcse/data/sampled_hotels.csv")
        
        if not os.path.exists(file_path):
            st.warning(f"酒店数据文件不存在: {file_path}")
            return pd.DataFrame()
            
        df = pd.read_csv(file_path)
        
        # 数据预处理
        # 1. 确保hotel_id为字符串类型
        df['hotel_id'] = df['hotel_id'].astype(str)
        
        # 2. 处理country_code列
        df['country_code'] = df['country_code'].fillna('Unknown')  # 将NaN替换为'Unknown'
        df['country_code'] = df['country_code'].astype(str)        # 转换为字符串类型
        
        # 3. 处理hotel_name列
        df['hotel_name'] = df['hotel_name'].fillna('未知酒店名称')   # 将NaN替换为默认值
        df['hotel_name'] = df['hotel_name'].astype(str)           # 转换为字符串类型
        
        return df
    except Exception as e:
        st.error(f"加载酒店数据文件失败: {str(e)}")
        logger.error(f"加载酒店数据文件失败: {str(e)}", exc_info=True)
        return pd.DataFrame()

# 加载酒店数据
HOTEL_DATA = load_hotel_data()

# 常量定义
DEFAULT_VALUES = {
    "hotel_id": HOTEL_DATA['hotel_id'].astype(str).tolist() if not HOTEL_DATA.empty else ["2560834","961935","806920"],
    "supplier_id": "0",
    "room_base_type": ["tourmind","ctrip","agoda","elong"],
    "room_status": "0",
    "cookie": "admintools_user_session=MTc0MTc0NTE3MnxHV0JIam9FOFFvenM3Tm9WUnlHUGZxWXVKeVFBRlZYVGxxdGdDNTRnMkp0M2hJeXBncFJIY0lBQk0wMjRXLXdqTVJjSGttbks5bzQ9fPKNVadREtRbtICRk6O1429iplB6Zu-AalkdjEEUUxxJ",
    "operator": "xiemingxuan",
    "referer": "http://erp.tourmind.cn/roomtypemapping/961935?supplierId=0&brt=tourmind&loadMaster=true",
    "threshold": 0.9,
    "model": "512batch_12e",
    "model_select": {
        "512batch_12e": "v2.1.0-beta (二月增强版)",
        "bert": "v1.0.0 (一月基础版)"
    },
    "match_api_url": "http://10.0.0.110:8100/match"
}

class RoomTypeAPI:
    """房型API接口类"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.logger = logging.getLogger(f"{__name__}.RoomTypeAPI")
    
    def fetch_room_data(
        self,
        hotel_id: str,
        supplier_id: str,
        room_base_type: str,
        room_status: str,
        cookie: str,
        operator: str,
        referer: str
    ) -> Optional[Dict]:
        """获取房型数据"""
        self.logger.debug(f"开始获取房型数据: hotel_id={hotel_id}, supplier_id={supplier_id}")
        
        params = {
            "sHotelID": hotel_id,
            "supplierID": supplier_id,
            "sRoomBaseType": room_base_type,
            "sroom_status": room_status
        }
        
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Cookie": cookie,
            "Operator": operator,
            "Referer": referer,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }
        
        try:
            self.logger.debug(f"发送请求到: {self.base_url}/roomtypemapping/api/get_roomtype")
            response = requests.get(
                f"{self.base_url}/roomtypemapping/api/get_roomtype",
                params=params,
                headers=headers,
                verify=False
            )
            self.logger.debug(f"API响应状态码: {response.status_code}")
            return response.json()
        except Exception as e:
            self.logger.error(f"获取数据时发生错误: {str(e)}，请检查个人的cookie是否过期，如有问题可联系同事：Maxon", exc_info=True)
            st.error(f"获取数据时发生错误: {str(e)}，请检查个人的cookie是否过期，如有问题可联系同事：Maxon")
            return None

class RoomTypeMatcher:
    """房型匹配器类"""
    
    def __init__(self, api_url: str, model_path: str = None):
        """初始化预测器"""
        self.default_api_url = api_url
        self.logger = logging.getLogger(f"{__name__}.RoomTypeMatcher")
        self.predictor = None
        
        # 初始化数据清洗器
        mapping_path = os.path.join(BASE_DIR, "match_by_simcse/data/mapping.xlsx")
        self.cleaner = DataCleaner(Path(mapping_path))
        
        # 初始化SimCSE预测器
        try:
            # 使用传入的model_path，如果为None则使用默认MODEL_PATH
            if not model_path:
                actual_model_path = MODEL_PATH
            else:
                # 判断是否是相对路径或仅模型名称
                if not os.path.isabs(model_path) and not model_path.startswith('/'):
                    # 构建完整路径，假设用户输入的是子目录名或模型名
                    if model_path.startswith("checkpoints_"):
                        # 如果以checkpoints_开头，认为是相对路径
                        actual_model_path = os.path.join(BASE_DIR, "match_by_simcse", model_path)
                    else:
                        # 如果只是模型名，则加到checkpoints_目录下
                        actual_model_path = os.path.join(BASE_DIR, "match_by_simcse/checkpoints_" + model_path)
                else:
                    # 已经是绝对路径
                    actual_model_path = model_path
            
            # 检查路径是否存在
            if not os.path.exists(actual_model_path):
                self.logger.warning(f"模型路径不存在: {actual_model_path}，尝试添加/best_model后缀")
                # 尝试添加best_model后缀
                if not actual_model_path.endswith('/best_model'):
                    actual_model_path = os.path.join(actual_model_path, 'best_model')
            
            self.logger.info(f"尝试加载模型: {actual_model_path}")
            try:
                self.predictor = SimCSEPredictor(actual_model_path)
                self.logger.info(f"成功初始化SimCSE预测器，使用模型: {actual_model_path}")
            except Exception as e:
                error_msg = f"加载自定义模型失败: {str(e)}"
                self.logger.warning(error_msg)
                st.warning(f"自定义模型 '{model_path}' 加载失败，尝试使用默认模型")
                
                # 尝试加载默认模型
                if actual_model_path != MODEL_PATH:
                    self.logger.info(f"尝试加载默认模型: {MODEL_PATH}")
                    try:
                        self.predictor = SimCSEPredictor(MODEL_PATH)
                        self.logger.info(f"成功初始化默认SimCSE预测器: {MODEL_PATH}")
                        st.success(f"已成功回退到默认模型: {os.path.basename(MODEL_PATH)}")
                    except Exception as fallback_error:
                        fallback_error_msg = f"默认模型加载也失败: {str(fallback_error)}"
                        self.logger.error(fallback_error_msg)
                        st.error(fallback_error_msg)
                else:
                    # 默认模型也加载失败
                    st.error(f"默认模型也无法加载: {str(e)}")
                
        except Exception as e:
            error_msg = f"初始化SimCSE预测器失败: {str(e)}"
            self.logger.error(error_msg)
            st.error(error_msg)
            self.logger.error(f"请确保模型路径正确，并且模型文件已下载到本地。在离线环境中，"
                           f"请参考 'https://huggingface.co/docs/transformers/installation#offline-mode' 配置离线模式。")
            st.error("如果在离线环境中运行，请确保已下载模型到本地，并正确配置离线模式。")
    
    def get_supplier_name(self, supplier_id: str) -> str:
        """获取供应商名称"""
        suppliers = SUPPLIER_MAPPING.get(supplier_id, ["未知"])
        return suppliers[0] if suppliers else "未知"
    
    def prepare_room_text(self, room: Dict) -> str:
        """准备房型文本用于匹配"""
        # 获取房型信息
        room_name = room.get('RoomTypeName', '')
        bed_type = room.get('BedTypeDesc', '')
        supplier_id = str(room.get('SupplierID', ''))
        
        # 获取供应商名称
        supplier_name = self.get_supplier_name(supplier_id)
        
        # 判断是否是供应商房型
        is_supplier = room.get('roomTypeListType') == 'SPLRoomTypeList'
        
        # 提取英文文本
        room_text = extract_english(room_name)
        bed_text = extract_english(bed_type)
        
        # 对供应商端数据进行清洗
        if is_supplier:
            if room_text:
                room_text = self.cleaner.clean_room_text(room_text)
            if bed_text:
                bed_text = self.cleaner.clean_bed_text(bed_text)
        
        # 组合成最终格式
        parts = []
        if room_text:
            parts.append(f"[room] {room_text}")
        if bed_text:
            parts.append(f"[bed] {bed_text}")
            
        # 转换为小写
        parts = [part.lower() for part in parts]
        return " ".join(filter(None, parts))
    
    def match_room_types(
        self,
        spl_rooms: List[Dict],
        s_rooms: List[Dict],
        threshold: float,
        model: str
    ) -> Optional[Dict]:
        """
        使用SimCSE模型进行房型匹配
        """
        try:
            if not self.predictor:
                raise Exception("SimCSE预测器未初始化")
            
            if not spl_rooms or not s_rooms:
                return None
            
            # 准备文本数据
            spl_texts = [self.prepare_room_text(room) for room in spl_rooms]
            s_texts = [self.prepare_room_text(room) for room in s_rooms]
            
            # 获取文本嵌入
            spl_embeddings = self.predictor.encode_batch(spl_texts)
            s_embeddings = self.predictor.encode_batch(s_texts)
            
            # 计算相似度
            similarities = self.predictor.calculate_similarity(spl_embeddings, s_embeddings)
            
            # 转换为numpy数组以便处理
            similarities_np = similarities.cpu().numpy()
            
            # 根据阈值生成预测结果
            predictions = (similarities_np > threshold).astype(float)
            
            # 构建返回结果
            results = []
            for i, spl_room in enumerate(spl_rooms):
                # 找出最佳匹配
                best_match_idx = similarities_np[i].argmax()
                best_similarity = similarities_np[i][best_match_idx]
                s_room = s_rooms[best_match_idx]
                
                if best_similarity >= threshold:
                    result = {
                        "SPL房型ID": spl_room.get("RoomTypeID", ""),
                        "标准房型ID": s_room.get("SRoomID", ""),
                        "供应商": self.get_supplier_name(str(spl_room.get("SupplierID", ""))),
                        "SPL房型": spl_room.get('RoomTypeName', ''),
                        "SPL床型": spl_room.get('BedTypeDesc', ''),
                        "标准房型": s_room.get('RoomTypeName', ''),
                        "标准床型": s_room.get('BedTypeDesc', ''),
                        "匹配分数": float(best_similarity),
                        "预测结果": 1 if best_similarity >= threshold else 0
                    }
                    results.append(result)
            
            return {
                "similarities": similarities_np.tolist(),
                "predictions": predictions.tolist(),
                "matches": results
            }
            
        except Exception as e:
            self.logger.error(f"房型匹配过程中发生错误: {str(e)}", exc_info=True)
            st.error(f"房型匹配过程中发生错误: {str(e)}")
            return None

class RoomTypeViewer:
    """房型匹配对比器主类"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RoomTypeViewer")
        self.room_api = RoomTypeAPI("http://erp.tourmind.cn")
        self.setup_page()
        
    def setup_page(self):
        """设置页面基本配置"""
        st.title("🏨 房型匹配对比器")
    
    def create_sidebar(self) -> Dict[str, Any]:
        """创建侧边栏并返回用户输入的参数"""
        with st.sidebar:
            st.header("🛠️ 参数设置")
            
            # 初始化session_state
            if 'hotel_id' not in st.session_state:
                st.session_state.hotel_id = DEFAULT_VALUES["hotel_id"][0]
            if 'is_custom_hotel_id' not in st.session_state:
                st.session_state.is_custom_hotel_id = False
            if 'selected_country' not in st.session_state:
                st.session_state.selected_country = 'ALL'
            if 'search_query' not in st.session_state:
                st.session_state.search_query = ''
            if 'match_api_url' not in st.session_state:
                st.session_state.match_api_url = DEFAULT_VALUES["match_api_url"]
            if 'is_batch_mode' not in st.session_state:
                st.session_state.is_batch_mode = False
            if 'max_hotels' not in st.session_state:
                st.session_state.max_hotels = 10
            if 'last_params' not in st.session_state:
                st.session_state.last_params = None
            if 'has_results' not in st.session_state:
                st.session_state.has_results = False
            
            # 处理模式选择
            is_batch_mode = st.checkbox(
                "批量处理模式", 
                value=st.session_state.is_batch_mode,
                help="开启后将批量处理选定国家的酒店",
                key="batch_mode_checkbox"
            )
            
            # 只在点击开始分析时更新session state
            if not st.session_state.has_results:
                st.session_state.is_batch_mode = is_batch_mode
            
            # 国家选择
            countries = ['ALL']
            if not HOTEL_DATA.empty:
                unique_countries = HOTEL_DATA['country_code'].unique()
                valid_countries = [c for c in unique_countries if c and c != 'Unknown']
                countries.extend(sorted(valid_countries))
                
            selected_country = st.selectbox(
                "选择国家",
                options=countries,
                index=countries.index(st.session_state.selected_country) if st.session_state.selected_country in countries else 0,
                help="选择要分析的酒店所在国家",
                key="country_selector"
            )
            
            # 只在点击开始分析时更新session state
            if not st.session_state.has_results:
                st.session_state.selected_country = selected_country
            
            # 酒店数量限制（批量模式）
            if st.session_state.is_batch_mode:
                max_hotels = st.number_input(
                    "最大处理酒店数",
                    min_value=1,
                    max_value=10000,
                    value=st.session_state.max_hotels,
                    help="限制批量处理时的最大酒店数量"
                )
                st.session_state.max_hotels = max_hotels

            # 添加搜索框，仅在非批量模式下显示
            if not st.session_state.is_batch_mode:
                search_query = st.text_input(
                    "搜索酒店",
                    value=st.session_state.search_query,
                    help="输入酒店ID或名称进行搜索"
                ).strip().lower()
                st.session_state.search_query = search_query
            
            # 根据选择的国家和搜索词筛选酒店列表
            filtered_hotels = HOTEL_DATA
            if not HOTEL_DATA.empty:
                # 应用国家筛选
                if selected_country != 'ALL':
                    filtered_hotels = filtered_hotels[filtered_hotels['country_code'] == selected_country]
                
                # 应用搜索筛选（仅在非批量模式下）
                if not st.session_state.is_batch_mode and st.session_state.search_query:
                    filtered_hotels = filtered_hotels[
                        filtered_hotels['hotel_id'].str.contains(st.session_state.search_query, case=False, na=False) |
                        filtered_hotels['hotel_name'].str.contains(st.session_state.search_query, case=False, na=False)
                    ]
            
            # 在批量模式下显示酒店数量
            if st.session_state.is_batch_mode:
                hotel_count = len(filtered_hotels) if not filtered_hotels.empty else 0
                st.info(f"将处理 {hotel_count} 家酒店")
                hotel_id = None  # 在批量模式下设置为None
            else:
                # 创建一个列来放置选择器和输入框
                col1, col2 = st.columns([3, 1])
                hotel_id = None  # 初始化hotel_id变量
                
                with col1:
                    if not st.session_state.is_custom_hotel_id:
                        # 使用selectbox选择预设的酒店ID，显示格式为: ID - 酒店名称
                        hotel_options = []
                        if not filtered_hotels.empty:
                            hotel_options = [
                                f"{row['hotel_id']} - {row['hotel_name']} ({row['country_code']})" 
                                for _, row in filtered_hotels.iterrows()
                            ]
                        else:
                            hotel_options = DEFAULT_VALUES["hotel_id"]
                        
                        if not hotel_options:  # 如果筛选后没有结果
                            st.warning("没有找到匹配的酒店")
                            hotel_options = ["无匹配结果"]
                        
                        selected_hotel = st.selectbox(
                            "选择酒店",
                            options=hotel_options,
                            index=0,
                            help="从列表中选择酒店"
                        )
                        
                        # 提取酒店ID
                        if selected_hotel != "无匹配结果":
                            hotel_id = selected_hotel.split(' - ')[0] if ' - ' in selected_hotel else selected_hotel
                    else:
                        # 使用text_input手动输入酒店ID
                        hotel_id = st.text_input(
                            "输入酒店ID",
                            value=st.session_state.hotel_id,
                            help="手动输入酒店ID"
                        )
                
                with col2:
                    # 添加切换按钮
                    if st.button("切换输入" if not st.session_state.is_custom_hotel_id else "使用列表"):
                        st.session_state.is_custom_hotel_id = not st.session_state.is_custom_hotel_id
                        st.rerun()
                
                # 更新session_state中的hotel_id
                if hotel_id:  # 只有在hotel_id有值时才更新
                    st.session_state.hotel_id = hotel_id
            
            # 如果选择了酒店，显示酒店详细信息
            if not st.session_state.is_batch_mode and hotel_id and not st.session_state.is_custom_hotel_id and not HOTEL_DATA.empty:
                hotel_info = HOTEL_DATA[HOTEL_DATA['hotel_id'] == hotel_id]
                if not hotel_info.empty:
                    hotel_info = hotel_info.iloc[0]
                    st.markdown("---")
                    st.markdown("#### 酒店信息")
                    st.markdown(f"**名称**: {hotel_info['hotel_name']}")
                    st.markdown(f"**国家**: {hotel_info['country_code']}")
                    st.markdown("---")
            
            # 模型选择部分
            st.subheader("模型选择")
            
            # 初始化模型选择的session state
            if 'is_custom_model' not in st.session_state:
                st.session_state.is_custom_model = False
            if 'custom_model' not in st.session_state:
                st.session_state.custom_model = ''
            
            # 获取可用的模型列表
            available_models = get_available_models()
            if available_models:
                st.info(f"当前环境中有 {len(available_models)} 个可用的预训练模型")
            else:
                st.warning("没有找到任何可用的预训练模型，请确保已下载模型到本地")
            
            # 创建一个列来放置选择器和切换按钮
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if not st.session_state.is_custom_model:
                    # 显示可用模型的信息
                    if available_models:
                        st.markdown("#### 可用的本地模型")
                        for name, path in available_models.items():
                            st.markdown(f"- **{name}**: `{path}`")
                    
                        # 使用selectbox选择已扫描到的可用模型
                        model_names = list(available_models.keys())
                        model_paths = list(available_models.values())
                        
                        # 创建模型路径到显示名称的映射
                        model_display_names = {
                            path: f"{name} - {get_model_display_name(path)}" 
                            for name, path in available_models.items()
                        }
                        
                        selected_model_path = st.selectbox(
                            "选择模型",
                            options=model_paths,
                            format_func=lambda x: model_display_names.get(x, os.path.basename(x)),
                            index=0,
                            help="选择要使用的模型"
                        )
                        # 使用模型路径作为main_model值
                        main_model = selected_model_path
                    else:
                        # 如果没有可用模型，显示警告并使用文本输入
                        st.warning("未检测到可用模型，请手动输入模型路径")
                        main_model = st.text_input(
                            "输入模型路径",
                            value=MODEL_PATH,
                            help="请输入一个有效的模型路径"
                        )
                else:
                    # 使用text_input手动输入模型名称
                    st.markdown("#### 自定义模型")
                    st.markdown("请输入以下格式之一:")
                    st.markdown("1. 相对路径: `checkpoints_xxx`")
                    st.markdown("2. 模型名: `xxx`（将自动添加checkpoints_前缀）")
                    st.markdown("3. 绝对路径: `/path/to/model`")
                    
                    main_model = st.text_input(
                        "输入模型名称",
                        value=st.session_state.custom_model,
                        help="手动输入模型名称或路径"
                    )
                    st.session_state.custom_model = main_model
            
            with col2:
                # 添加切换按钮
                if st.button(
                    "切换输入" if not st.session_state.is_custom_model else "使用列表",
                    key="toggle_model_input"  # 添加唯一的key
                ):
                    st.session_state.is_custom_model = not st.session_state.is_custom_model
                    st.rerun()
            
            # 修改match_api_url的处理
            input_api_url = st.text_input(
                "match_url",
                value=st.session_state.match_api_url,
                help="设置匹配服务的API地址",
                key="match_api_url_input"
            )
            # 更新session_state中的match_api_url
            if input_api_url != st.session_state.match_api_url:
                st.session_state.match_api_url = input_api_url
            
            # 设置参数
            if st.session_state.is_batch_mode:
                # 批量模式下，使用过滤后的酒店列表
                hotel_ids = filtered_hotels['hotel_id'].tolist() if not filtered_hotels.empty else []
                params = {
                    "hotel_ids": hotel_ids,
                    "is_batch_mode": True,
                    "filtered_hotels": filtered_hotels,
                    "max_hotels": st.session_state.max_hotels
                }
            else:
                # 单酒店模式
                params = {
                    "hotel_id": st.session_state.hotel_id,
                    "is_batch_mode": False
                }
                
            # 通用参数
            params.update({
                "room_base_type": st.selectbox("房型基础类型", DEFAULT_VALUES["room_base_type"], index=0, help="选择房型的基础类型"),
                "model": main_model,
                "threshold": st.slider("匹配阈值", min_value=0.7, max_value=1.0, value=DEFAULT_VALUES["threshold"], 
                                    step=0.01, help="设置房型匹配的阈值，值越大匹配越严格"),
                "match_api_url": input_api_url,
                "supplier_id": DEFAULT_VALUES["supplier_id"],
                "room_status": DEFAULT_VALUES["room_status"]
            })
            
            st.header("高级参数设置")
            with st.expander("API认证信息"):
                params.update({
                    "cookie": st.text_input("Cookie", value=DEFAULT_VALUES["cookie"], 
                                          help="API认证Cookie", type="password"),
                    "operator": st.text_input("操作员", value=DEFAULT_VALUES["operator"], 
                                            help="操作员名称"),
                    "referer": DEFAULT_VALUES["referer"]
                })
            
            # 添加查询按钮
            query_button = st.button("开始分析", use_container_width=True, key="start_analysis")
            
            # 如果点击了开始分析按钮
            if query_button:
                st.session_state.has_results = False
                st.session_state.last_params = None
                params = {
                    "hotel_ids": hotel_ids if st.session_state.is_batch_mode else None,
                    "is_batch_mode": st.session_state.is_batch_mode,
                    "filtered_hotels": filtered_hotels if st.session_state.is_batch_mode else None,
                    "max_hotels": st.session_state.max_hotels if st.session_state.is_batch_mode else None,
                    "hotel_id": st.session_state.hotel_id if not st.session_state.is_batch_mode else None,
                    "room_base_type": params["room_base_type"],
                    "model": params["model"],
                    "threshold": params["threshold"],
                    "match_api_url": params["match_api_url"],
                    "supplier_id": params["supplier_id"],
                    "room_status": params["room_status"],
                    "cookie": params["cookie"],
                    "operator": params["operator"],
                    "referer": params["referer"],
                    "query_button": True
                }
                st.session_state.last_params = params
                return params
            
            # 如果已经有结果且没有点击开始分析，返回上次的参数
            if st.session_state.has_results and st.session_state.last_params:
                return st.session_state.last_params
            
            # 默认返回当前参数但不触发查询
            return {
                "query_button": False,
                "is_batch_mode": st.session_state.is_batch_mode,
                "hotel_id": st.session_state.hotel_id,
                # ... other parameters ...
            }
    
    def process_room_data(self, data: Dict) -> tuple[pd.DataFrame, List[Dict], List[Dict]]:
        """处理房型数据，返回DataFrame和分类后的房型列表"""
        # 获取房型列表
        hotel_data = data.get("Data", {})
        spl_rooms = hotel_data.get("SPLRoomTypeList", [])
        s_rooms = hotel_data.get("SRoomTypeList", [])
        
        # 创建包含所有房型的列表
        all_rooms = []
        
        # 处理 SPL 房型
        for room in spl_rooms:
            room["roomTypeListType"] = "SPLRoomTypeList"
            all_rooms.append(room)
            
        # 处理标准房型
        for room in s_rooms:
            room["roomTypeListType"] = "SRoomTypeList"
            all_rooms.append(room)
        
        # 转换为 DataFrame
        df = pd.DataFrame(all_rooms)
        
        return df, spl_rooms, s_rooms
    
    def show_data_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """显示并应用数据筛选器"""
        # 不显示筛选器UI，但保留筛选功能
        status_filter = None
        name_filter = None
        type_filter = None
        
        # 应用筛选
        if status_filter:
            df = df[df["status"].isin(status_filter)]
        if name_filter:
            df = df[df["roomTypeName"].str.contains(name_filter, case=False, na=False)]
        if type_filter:
            df = df[df["roomTypeListType"].isin(type_filter)]
            
        return df
    
    def show_statistics(self, df: pd.DataFrame, spl_rooms: List[Dict], s_rooms: List[Dict]):
        """显示数据统计信息"""
        st.subheader("数据统计")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("SPL房型数", len(spl_rooms))
        with col2:
            st.metric("标准房型数", len(s_rooms))

    
    def process_matches(self, match_results: List[Dict], spl_rooms: List[Dict], s_rooms: List[Dict], model: str) -> Optional[pd.DataFrame]:
        """处理匹配结果，并对比原有匹配"""
        if not match_results:
            return None
            
        # 处理主要模型的结果
        match_df = pd.DataFrame(match_results)
        match_df = match_df.sort_values(by="匹配分数", ascending=False)
        
        # 创建标准房型ID到房型信息的映射
        s_room_map = {
            str(room.get("SRoomID", "")): {
                "房型名称": room.get("RoomTypeName", ""),
                "房型中文": room.get("RoomTypeNameCN", ""),
                "床型": room.get("BedTypeDesc", ""),
                "床型中文": room.get("BedTypeDescCN", "")
            }
            for room in s_rooms
        }
        
        # 创建一个映射来存储原有的匹配关系
        original_matches = {
            room["RoomTypeID"]: {
                "SPL房型": room["RoomTypeName"],
                "SPL房型中文": room["RoomTypeNameCN"],
                "SPL床型": room["BedTypeDesc"],
                "SPL床型中文": room["BedTypeDescCN"],
                "规则匹配ID": room["SRoomID"],
                "规则匹配分数": room.get("Score", 0),
                "状态": room["Status"]
            }
            for room in spl_rooms
        }
        
        # 添加对比结果列
        def get_match_status(row):
            for room_id, room_info in original_matches.items():
                if room_id != row["SPL房型ID"]:
                    continue
                if (room_info["规则匹配ID"] ==0):
                    return "原未匹配"
                if (room_info["规则匹配ID"] >0) and (row["标准房型ID"]==room_info["规则匹配ID"]):
                    return "已匹配" 
                if (room_info["规则匹配ID"] >0) and (row["标准房型ID"]!=room_info["规则匹配ID"]):
                    return "匹配变化"
            return "未知"
            
        match_df["匹配状态"] = match_df.apply(get_match_status, axis=1)
        
        # 统计原有匹配的房型数量
        original_matched_count = sum(1 for room in spl_rooms if room["SRoomID"] >0)
        total_spl_rooms = len(spl_rooms)
        
        # 计算各种匹配状态的数量
        new_match_count = len(match_df)  # 新的匹配总数
        originally_matched = len(match_df[match_df["匹配状态"] == "已匹配"])  # 原本就匹配的数量
        newly_matched = len(match_df[match_df["匹配状态"] == "原未匹配"])  # 新增匹配的数量
        lost_matches = len(match_df[match_df["匹配状态"] == "匹配变化"])  # 匹配丢失的数量
        
        # 显示匹配统计
        st.subheader("匹配统计")
        
        # 显示基本统计信息
        col1, col2 = st.columns(2)

        with col1:
            st.metric("规则匹配数", original_matched_count)
        with col2:
            st.metric("规则未匹配数", total_spl_rooms-original_matched_count)
            
        # 显示模型统计
        st.markdown(f"##### 模型 ({model}) 匹配结果")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("模型匹配数", new_match_count)
        with col2:
            st.metric("模型未匹配数", total_spl_rooms-new_match_count)
        with col3:
            st.metric("保持匹配数", originally_matched)
        with col4:
            # 计算模型新增匹配率：新增匹配数 / 规则未匹配数
            unmatched_count = total_spl_rooms - original_matched_count
            new_match_rate = (newly_matched / unmatched_count * 100) if unmatched_count > 0 else 0
            st.metric("模型新增匹配率", f"{new_match_rate:.2f}%", 
                    help="模型新增匹配数占规则未匹配数的比例")
            
        # 显示模型的详细变化统计
        st.markdown("##### 匹配变化详情")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("保持匹配数", originally_matched, 
                     help="原本匹配且现在仍然匹配的房型数量")
        with col2:
            st.metric("新增匹配数", newly_matched,
                     help="原本未匹配但现在匹配上的房型数量")
        with col3:
            st.metric("匹配变化数", lost_matches,
                     help="原本匹配但现在使用模型匹配后，匹配对象发生变化")
        
        # 显示匹配状态变化
        st.subheader("匹配状态变化")
        
        # 显示新增的匹配
        if newly_matched > 0:
            st.markdown("##### 🆕 新增匹配")
            new_matches_df = match_df[match_df["匹配状态"] == "原未匹配"]
            st.dataframe(new_matches_df, use_container_width=True)
        
        # 显示匹配变化
        if lost_matches > 0:
            st.markdown("##### ❌ 匹配变化")
            lost_matches_info = []
            
            # 获取所有匹配状态为"匹配变化"的记录
            changed_matches = match_df[match_df["匹配状态"] == "匹配变化"]
            
            for _, row in changed_matches.iterrows():
                spl_room_id = row["SPL房型ID"]
                room_info = original_matches.get(spl_room_id, {})
                
                if room_info:
                    # 获取规则匹配的标准房型信息
                    original_s_room = s_room_map.get(str(room_info["规则匹配ID"]), {})
                    
                    # 获取规则匹配分数
                    original_match_score = 0
                    # 找到对应的SPL房型
                    spl_room = next((room for room in spl_rooms if str(room.get("RoomTypeID")) == str(spl_room_id)), None)
                    if spl_room:
                        # 重新调用匹配API获取相似度
                        match_result = self.matcher.match_room_types(
                            [spl_room],
                            s_rooms,
                            0.0,  # 使用0作为阈值，确保能获取到所有相似度
                            model
                        )
                        if match_result and "similarities" in match_result:
                            # 找到规则匹配房型在s_rooms中的索引
                            original_idx = next(
                                (idx for idx, room in enumerate(s_rooms) 
                                 if str(room.get("SRoomID")) == str(room_info["规则匹配ID"])),
                                None
                            )
                            if original_idx is not None:
                                original_match_score = match_result["similarities"][original_idx]

                    lost_info = {
                        "SPL房型ID": str(row["SPL房型ID"]),
                        "规则匹配ID": str(room_info["规则匹配ID"]),
                        "模型匹配ID": str(row["标准房型ID"]),
                        "规则匹配分数": original_match_score,
                        "规则匹配房型": original_s_room.get('房型名称', '未知'),
                        "规则匹配床型": original_s_room.get('床型', '未知'),
                        "模型匹配房型": row['标准房型'],
                        "模型匹配床型": row['标准床型'],
                        "模型匹配分数": row["匹配分数"]
                    }
                    lost_matches_info.append(lost_info)
            
            if lost_matches_info:
                lost_df = pd.DataFrame(lost_matches_info)
                
                # 确保所有必需的列都存在
                required_columns = [
                    "SPL房型ID", "规则匹配ID" ,"模型匹配ID", 
                    "规则匹配分数", "规则匹配房型", "规则匹配床型",
                    "SPL房型", "SPL床型", 
                    "模型匹配房型", "模型匹配床型", "模型匹配分数"
                ]
                
                # 检查是否有缺失的列，如果有则添加默认值
                for col in required_columns:
                    if col not in lost_df.columns:
                        lost_df[col] = "未知"
                
                # 重新排列列顺序
                lost_df = lost_df[required_columns]
                
                # 显示DataFrame
                st.dataframe(lost_df, use_container_width=True)
                
                # 添加导出功能
                st.download_button(
                    label="导出匹配变化数据",
                    data=lost_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'匹配变化数据_{DEFAULT_VALUES["hotel_id"]}.csv',
                    mime='text/csv',
                    key=f"download_lost_{model}"
                )
            else:
                st.info("没有匹配变化的记录")
        
        return match_df
    
    def run(self):
        """运行房型查看器"""
        params = self.create_sidebar()
        
        # 只在点击开始分析按钮时执行查询
        if params["query_button"]:
            # 批量模式
            if params["is_batch_mode"]:
                self.run_batch_mode(params)
            # 单酒店模式
            else:
                self.run_single_mode(params)
            # 设置已有结果标志
            st.session_state.has_results = True
    
    def run_single_mode(self, params):
        """单酒店处理模式"""
        try:
            # 创建进度条和状态文本
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 创建结果展示区域
            hotel_info_container = st.container()
            stats_container = st.container()
            results_container = st.container()

            with stats_container:
                st.header("批量处理汇总结果")
                
                # 创建统计信息的占位符
                col1, col2 = st.columns(2)
                with col1:
                    total_hotels_metric = st.empty()
                    success_hotels_metric = st.empty()
                    failed_hotels_metric = st.empty()
                with col2:
                    total_rooms_metric = st.empty()
                    matched_rooms_metric = st.empty()
                    unmatched_rooms_metric = st.empty()
                    match_rate_metric = st.empty()
                
                # 创建模型匹配结果的占位符
                st.markdown(f"##### 模型 ({params['model']}) 匹配结果")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    model_matched_metric = st.empty()
                with col2:
                    model_unmatched_metric = st.empty()
                with col3:
                    kept_matched_metric = st.empty()
                with col4:
                    changed_matched_metric = st.empty()
                
                # 创建匹配变化详情的占位符
                st.markdown("##### 匹配变化详情")
                col1, col2, col3 = st.columns(3)
                with col1:
                    kept_detail_metric = st.empty()
                with col2:
                    new_detail_metric = st.empty()
                with col3:
                    changed_detail_metric = st.empty()

            with results_container:
                # 创建结果展示的占位符
                new_matches_title = st.empty()
                new_matches_df_placeholder = st.empty()
                new_download_btn = st.empty()
                
                changed_matches_title = st.empty()
                changed_matches_df_placeholder = st.empty()
                changed_download_btn = st.empty()
                
                unmatched_title = st.empty()
                unmatched_df_placeholder = st.empty()
                unmatched_download_btn = st.empty()
                
                all_download_btn = st.empty()

            # ... existing code ...

            # 获取酒店名称（如果在HOTEL_DATA中有记录）
            hotel_name = "未知酒店"
            if not HOTEL_DATA.empty:
                hotel_info = HOTEL_DATA[HOTEL_DATA['hotel_id'] == params["hotel_id"]]
                if not hotel_info.empty:
                    hotel_name = hotel_info.iloc[0]['hotel_name']
            
            # 展示当前正在处理的酒店信息
            with hotel_info_container:
                st.markdown(f"### 🏨 正在处理酒店: {hotel_name} (ID: {params['hotel_id']})")
            
            status_text.text(f"📡 正在获取酒店数据... ({params['hotel_id']})")
            progress_bar.progress(0.1)
            
            data = self.room_api.fetch_room_data(
                params["hotel_id"],
                params["supplier_id"],
                params["room_base_type"],
                params["room_status"],
                params["cookie"],
                params["operator"],
                params["referer"]
            )
            
            if data:
                progress_bar.progress(0.3)
                status_text.text("🔍 正在解析房型数据...")
                
                if isinstance(data, dict) and "Data" in data:
                    # 处理数据
                    df, spl_rooms, s_rooms = self.process_room_data(data)
                    progress_bar.progress(0.4)
                    
                    # 立即显示房型数量统计
                    with stats_container:
                        st.markdown("### 📊 房型数量统计")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("SPL房型数量", len(spl_rooms), help="供应商提供的房型数量")
                        with col2:
                            st.metric("标准房型数量", len(s_rooms), help="系统中的标准房型数量")
                    
                    status_text.text("🧮 正在分析房型匹配关系...")
                    progress_bar.progress(0.6)
                    
                    with results_container:
                        if not df.empty:
                            df = self.show_data_filters(df)
                            
                            # 显示更多统计信息
                            progress_bar.progress(0.7)
                            status_text.text("📈 正在生成匹配统计...")
                            
                            # 添加匹配功能
                            if spl_rooms and s_rooms:
                                progress_bar.progress(0.8)
                                status_text.text("🔄 正在执行房型匹配分析...")
                                self.run_matching(spl_rooms, s_rooms, params["threshold"], params["model"])
                            
                            progress_bar.progress(0.9)
                            status_text.text("📋 正在生成最终报告...")
                            
                            # 添加数据导出功能
                            st.download_button(
                                label="📥 导出原始数据",
                                data=df.to_csv(index=False).encode('utf-8'),
                                file_name=f'房型数据_{params["hotel_id"]}.csv',
                                mime='text/csv',
                                key="download_original_data"
                            )
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ 分析完成！房型匹配报告已生成。")
                else:
                    st.warning("未找到有效的房型数据")
                    status_text.text("⚠️ 未找到有效的房型数据")
            else:
                st.error("获取酒店数据失败")
                status_text.text("❌ 获取酒店数据失败")
        except Exception as e:
            st.error(f"处理数据时发生错误: {str(e)}")
            status_text.text("❌ 处理失败！")
            self.logger.error(f"处理数据时发生错误: {str(e)}", exc_info=True)
    
    def run_batch_mode(self, params):
        """批量处理模式"""
        hotel_ids = params["hotel_ids"]
        
        if not hotel_ids:
            st.warning("没有找到需要处理的酒店")
            return
        

        #  # 限制处理的酒店数量
        # if len(hotel_ids) > params.get("max_hotels", 10):
        #     st.info(f"已限制处理前 {params['max_hotels']} 家酒店（共 {len(hotel_ids)} 家）")
        #     hotel_ids = hotel_ids[:params["max_hotels"]]

        # 使用pandas的sample方法随机选择酒店
        filtered_hotels = params["filtered_hotels"]
        total_hotels = len(hotel_ids)
        max_hotels = params.get("max_hotels", 10)
        
        if total_hotels > max_hotels:
            st.info(f"将随机选择 {max_hotels} 家酒店进行处理（共 {total_hotels} 家）")
            # 随机选择酒店
            selected_hotels = filtered_hotels.sample(n=max_hotels, random_state=42)  # 设置random_state以保证结果可重现
            hotel_ids = selected_hotels['hotel_id'].tolist()
        else:
            st.info(f"将处理全部 {total_hotels} 家酒店")
            selected_hotels = filtered_hotels
        
        # 创建进度条和状态文本
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 创建当前酒店信息展示区
        current_hotel_container = st.container()
        with current_hotel_container:
            st.subheader("🏨 当前处理酒店")
            current_hotel_info = st.empty()
            # 为当前酒店创建进度条和状态文本
            current_hotel_progress = st.progress(0)
            current_hotel_status = st.empty()
            # 创建当前酒店的房型数量统计
            current_hotel_stats = st.empty()
        
        # 创建结果展示区域的占位符
        stats_container = st.container()
        chart_container = st.container()
        results_container = st.container()

        with stats_container:
            st.header("批量处理汇总结果")
            
            # 创建统计信息的占位符
            col1, col2 = st.columns(2)
            with col1:
                total_hotels_metric = st.empty()
                success_hotels_metric = st.empty()
                failed_hotels_metric = st.empty()
            with col2:
                total_rooms_metric = st.empty()
                matched_rooms_metric = st.empty()
                unmatched_rooms_metric = st.empty()
                match_rate_metric = st.empty()
            
            # 创建模型匹配结果的占位符
            st.markdown(f"##### 模型 ({params['model']}) 匹配结果")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                model_matched_metric = st.empty()
            with col2:
                model_unmatched_metric = st.empty()
            with col3:
                kept_matched_metric = st.empty()
            with col4:
                changed_matched_metric = st.empty()
            
            # 创建匹配变化详情的占位符
            st.markdown("##### 匹配变化详情")
            col1, col2, col3 = st.columns(3)
            with col1:
                kept_detail_metric = st.empty()
            with col2:
                new_detail_metric = st.empty()
            with col3:
                changed_detail_metric = st.empty()

        with chart_container:
            chart_placeholder = st.empty()
            
        with results_container:
            # 创建结果展示的占位符
            new_matches_title = st.empty()
            new_matches_df_placeholder = st.empty()
            new_download_btn = st.empty()
            
            changed_matches_title = st.empty()
            changed_matches_df_placeholder = st.empty()
            changed_download_btn = st.empty()
            
            unmatched_title = st.empty()
            unmatched_df_placeholder = st.empty()
            unmatched_download_btn = st.empty()
            
            all_download_btn = st.empty()

        # 汇总结果
        all_results = []
        all_unmatched = []
        all_changed = []
        all_newly_matched = []
        
        # 统计信息
        total_hotels = len(hotel_ids)
        processed_hotels = 0
        success_hotels = 0
        failed_hotels = 0
        total_spl_rooms = 0
        total_matched = 0
        total_unmatched = 0
        total_changed = 0
        total_newly = 0
        
        for i, hotel_id in enumerate(hotel_ids):
            status_text.text(f"处理酒店 {hotel_id} ({i+1}/{total_hotels})...")
            
            try:
                # 获取酒店信息
                hotel_info = params["filtered_hotels"][params["filtered_hotels"]["hotel_id"] == hotel_id]
                hotel_name = hotel_info["hotel_name"].values[0] if not hotel_info.empty else "未知酒店"
                country_code = hotel_info["country_code"].values[0] if not hotel_info.empty else "未知"
                
                # 更新当前酒店信息
                current_hotel_info.markdown(f"### 当前处理: {hotel_name} (ID: {hotel_id}, 国家: {country_code})")
                current_hotel_progress.progress(0)  # 重置当前酒店进度条
                current_hotel_status.text("📡 正在获取酒店数据...")
                
                # 获取数据
                data = self.room_api.fetch_room_data(
                    hotel_id,
                    params["supplier_id"],
                    params["room_base_type"],
                    params["room_status"],
                    params["cookie"],
                    params["operator"],
                    params["referer"]
                )
                
                if data and isinstance(data, dict) and "Data" in data:
                    # 处理数据
                    df, spl_rooms, s_rooms = self.process_room_data(data)
                    
                    current_hotel_progress.progress(0.3)
                    current_hotel_status.text("🔍 正在解析房型数据...")
                    
                    if spl_rooms and s_rooms:
                        # 创建matcher实例
                        self.matcher = RoomTypeMatcher(params["match_api_url"], params["model"])
                        
                        # 匹配结果
                        match_results = []
                        unmatched_results = []
                        
                        # 运行模型匹配
                        total_spl = len(spl_rooms)
                        for spl_idx, spl_room in enumerate(spl_rooms):
                            # 更新当前房型匹配进度
                            sub_progress = (1 * (spl_idx + 1) / total_spl)
                            current_hotel_progress.progress(sub_progress)
                            current_hotel_status.text(f"🔄 正在匹配房型 ({spl_idx+1}/{total_spl})...")
                            
                            match_result = self.matcher.match_room_types(
                                [spl_room], 
                                s_rooms, 
                                params["threshold"],
                                params["model"]
                            )
                            
                            if match_result and "similarities" in match_result:
                                self._process_single_match(match_result, spl_room, s_rooms, match_results, unmatched_results)
                        
                        current_hotel_progress.progress(0.9)
                        current_hotel_status.text("📊 正在生成匹配报告...")
                        
                        # 为每个结果添加酒店信息
                        for result in match_results:
                            result["酒店ID"] = hotel_id
                            result["酒店名称"] = hotel_name
                            result["国家"] = country_code
                        
                        for result in unmatched_results:
                            result["酒店ID"] = hotel_id
                            result["酒店名称"] = hotel_name
                            result["国家"] = country_code
                        
                        # 处理匹配状态
                        if match_results:
                            match_df = pd.DataFrame(match_results)
                            
                            # 创建一个映射来存储原有的匹配关系
                            original_matches = {
                                room["RoomTypeID"]: {
                                    "规则匹配ID": room["SRoomID"],
                                    "状态": room["Status"]
                                }
                                for room in spl_rooms
                            }
                            
                            # 添加匹配状态
                            def get_match_status(row):
                                room_id = row["SPL房型ID"]
                                room_info = original_matches.get(room_id, {})
                                
                                if not room_info:
                                    return "未知"
                                    
                                if room_info.get("规则匹配ID", 0) == 0:
                                    return "原未匹配"
                                if (room_info.get("规则匹配ID", 0) > 0) and (row["标准房型ID"] == room_info.get("规则匹配ID")):
                                    return "已匹配" 
                                if (room_info.get("规则匹配ID", 0) > 0) and (row["标准房型ID"] != room_info.get("规则匹配ID")):
                                    return "匹配变化"
                                    
                                return "未知"
                            
                            match_df["匹配状态"] = match_df.apply(get_match_status, axis=1)
                            
                            # 统计
                            original_matched_count = sum(1 for room in spl_rooms if room["SRoomID"] > 0)
                            new_match_count = len(match_df)
                            originally_matched = len(match_df[match_df["匹配状态"] == "已匹配"])
                            newly_matched = len(match_df[match_df["匹配状态"] == "原未匹配"])
                            lost_matches = len(match_df[match_df["匹配状态"] == "匹配变化"])
                            
                            # 完成当前酒店匹配状态展示
                            current_hotel_status.text(f"✅ 匹配完成! 原规则匹配: {original_matched_count}, 模型匹配: {new_match_count}, 新增: {newly_matched}, 变化: {lost_matches}")
                            current_hotel_progress.progress(1.0)
                            
                            # 更新汇总统计 - 修改这部分逻辑
                            current_total = len(spl_rooms)  # 当前酒店的总房型数
                            current_matched = original_matched_count  # 当前酒店的规则匹配数
                            current_unmatched = current_total - current_matched  # 当前酒店的规则未匹配数
                            
                            total_spl_rooms += current_total
                            total_matched += current_matched
                            total_unmatched += current_unmatched
                            total_changed += lost_matches
                            total_newly += newly_matched
                            
                            # 提取新增匹配和变化匹配
                            new_matches = match_df[match_df["匹配状态"] == "原未匹配"]
                            changed_matches = match_df[match_df["匹配状态"] == "匹配变化"]
                            
                            if not new_matches.empty:
                                all_newly_matched.append(new_matches)
                            
                            if not changed_matches.empty:
                                all_changed.append(changed_matches)
                            
                            # 添加到汇总结果
                            all_results.append(match_df)
                        
                        if unmatched_results:
                            all_unmatched.append(pd.DataFrame(unmatched_results))
                        
                        success_hotels += 1
                    else:
                        current_hotel_status.text("❌ 该酒店没有可用的房型数据")
                        current_hotel_progress.progress(1.0)
                        self.logger.warning(f"酒店 {hotel_id} 没有房型数据")
                else:
                    current_hotel_status.text("❌ 获取酒店数据失败")
                    current_hotel_progress.progress(1.0)
                    self.logger.warning(f"获取酒店 {hotel_id} 数据失败")
                    failed_hotels += 1
            except Exception as e:
                current_hotel_status.text(f"❌ 处理失败: {str(e)}")
                current_hotel_progress.progress(1.0)
                self.logger.error(f"处理酒店 {hotel_id} 时发生错误: {str(e)}", exc_info=True)
                failed_hotels += 1
            
            # 更新进度
            processed_hotels += 1
            progress_bar.progress((i + 1) / total_hotels)
            
            # 实时更新统计信息
            # 使用占位符更新统计信息
            total_hotels_metric.metric("总酒店数", total_hotels)
            success_hotels_metric.metric("处理成功", success_hotels)
            failed_hotels_metric.metric("处理失败", failed_hotels)
            
            total_rooms_metric.metric("总房型数", total_spl_rooms)
            matched_rooms_metric.metric("规则匹配数", total_matched)
            unmatched_rooms_metric.metric("规则未匹配数", total_unmatched)
            match_rate_metric.metric("规则匹配率", f"{(total_matched/total_spl_rooms*100):.2f}%" if total_spl_rooms > 0 else "0%")
            
            # 更新模型匹配结果
            model_matched_metric.metric("模型匹配数", total_matched + total_newly)
            model_unmatched_metric.metric("模型未匹配数", total_spl_rooms - (total_matched + total_newly))
            kept_matched_metric.metric("保持匹配数", total_matched - total_changed)
            # 计算模型新增匹配率：新增匹配数 / 规则未匹配数
            new_match_rate = (total_newly / total_unmatched * 100) if total_unmatched > 0 else 0
            changed_matched_metric.metric("模型新增匹配率", f"{new_match_rate:.2f}%", 
                                        help="模型新增匹配数占规则未匹配数的比例")
            
            # 更新匹配变化详情
            kept_detail_metric.metric("保持匹配数", total_matched - total_changed,
                         help="原本匹配且现在仍然匹配的房型数量")
            new_detail_metric.metric("新增匹配数", total_newly,
                         help="原本未匹配但现在匹配上的房型数量")
            changed_detail_metric.metric("匹配变化数", total_changed,
                         help="原本匹配但现在使用模型匹配后，匹配对象发生变化")
            
            # 显示匹配状态分布图表
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                status_counts = combined_results["匹配状态"].value_counts()
                with chart_container:
                    chart_placeholder.bar_chart(status_counts)
            
            # 更新新增匹配显示
            if all_newly_matched:
                new_matches_title.markdown("### 🆕 新增匹配")
                new_matches_df = pd.concat(all_newly_matched, ignore_index=True).drop_duplicates()
                new_matches_df_placeholder.dataframe(new_matches_df, use_container_width=True)
                
                new_download_btn.download_button(
                    label="📥 导出新增匹配数据",
                    data=new_matches_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'新增匹配数据_{st.session_state.selected_country}.csv',
                    mime='text/csv',
                    key=f"download_new_{i}"
                )
            else:
                new_matches_title.empty()
                new_matches_df_placeholder.info("暂无新增匹配")
                new_download_btn.empty()
            
            # 更新匹配变化显示
            if all_changed:
                changed_matches_title.markdown("### ❌ 匹配变化")
                changed_matches_df = pd.concat(all_changed, ignore_index=True).drop_duplicates()
                changed_matches_df_placeholder.dataframe(changed_matches_df, use_container_width=True)
                
                changed_download_btn.download_button(
                    label="📥 导出匹配变化数据",
                    data=changed_matches_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'匹配变化数据_{st.session_state.selected_country}.csv',
                    mime='text/csv',
                    key=f"download_changed_{i}"
                )
            else:
                changed_matches_title.empty()
                changed_matches_df_placeholder.info("暂无匹配变化")
                changed_download_btn.empty()
            
            # 更新未匹配结果显示
            if all_unmatched:
                unmatched_title.markdown("### ⚠️ 未匹配房型")
                unmatched_df = pd.concat(all_unmatched, ignore_index=True).drop_duplicates()
                unmatched_df_placeholder.dataframe(unmatched_df, use_container_width=True)
                
                unmatched_download_btn.download_button(
                    label="📥 导出未匹配数据",
                    data=unmatched_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'未匹配数据_{st.session_state.selected_country}.csv',
                    mime='text/csv',
                    key=f"download_unmatched_{i}"
                )
            else:
                unmatched_title.empty()
                unmatched_df_placeholder.info("暂无未匹配房型")
                unmatched_download_btn.empty()
            
            # 更新全部结果下载按钮
            if all_results:
                all_results_df = pd.concat(all_results, ignore_index=True).drop_duplicates()
                all_download_btn.download_button(
                    label="📥 导出全部匹配结果",
                    data=all_results_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'全部匹配结果_{st.session_state.selected_country}.csv',
                    mime='text/csv',
                    key=f"download_all_{i}"
                )
            else:
                all_download_btn.empty()
        
        # 完成处理
        status_text.text(f"✅ 批量处理完成！共处理 {processed_hotels} 家酒店，成功 {success_hotels} 家，失败 {failed_hotels} 家。")

    def _process_single_match(self, match_result: Dict, spl_room: Dict, s_rooms: List[Dict], results: List[Dict], unmatched_results: List[Dict]):
        """处理单个房型的匹配结果"""
        similarities = match_result["similarities"]
        predictions = match_result.get("predictions", [])
        
        # 确保similarities是一维数组
        if isinstance(similarities[0], list):
            similarities = similarities[0]  # 如果是二维数组，取第一行
        
        # 找出最高相似度的索引和分数
        best_match_idx = max(range(len(similarities)), key=lambda i: similarities[i])
        best_similarity = similarities[best_match_idx]
        s_room = s_rooms[best_match_idx]
        
        # 确保predictions也是一维数组
        if isinstance(predictions[0], list):
            predictions = predictions[0]
        
        # 构建匹配信息
        match_info = {
            "SPL房型ID": spl_room.get("RoomTypeID", ""),
            "标准房型ID": s_room.get("SRoomID", ""),
            "供应商": self.matcher.get_supplier_name(str(spl_room.get("SupplierID", ""))),
            "SPL房型": spl_room.get('RoomTypeName', ''),
            "SPL床型": spl_room.get('BedTypeDesc', ''),
            "标准房型": s_room.get('RoomTypeName', ''),
            "标准床型": s_room.get('BedTypeDesc', ''),
            "匹配分数": float(best_similarity),
            "预测结果": 1 if predictions[best_match_idx] == 1 else 0
        }
        
        # 根据预测结果分类
        if predictions[best_match_idx] == 1:
            results.append(match_info)
        elif unmatched_results is not None:
            unmatched_results.append(match_info)

    def run_matching(self, spl_rooms: List[Dict], s_rooms: List[Dict], threshold: float, model: str):
        """运行房型匹配"""
        try:
            if not spl_rooms or not s_rooms:
                st.warning("没有可匹配的房型数据")
                return
                
            # 创建matcher实例
            self.matcher = RoomTypeMatcher(st.session_state.match_api_url, model)
            
            # 匹配结果
            match_results = []
            unmatched_results = []
            
            # 显示匹配进度
            match_progress = st.progress(0)
            match_status = st.empty()
            
            # 运行模型匹配
            match_status.text("🔍 正在匹配房型...")
            total_spl_rooms = len(spl_rooms)
            
            for i, spl_room in enumerate(spl_rooms):
                match_progress.progress((i + 1) / total_spl_rooms)
                match_status.text(f"🔍 正在匹配房型 ({i+1}/{total_spl_rooms})...")
                
                match_result = self.matcher.match_room_types(
                    [spl_room], 
                    s_rooms, 
                    threshold,
                    model
                )
                
                if match_result and "similarities" in match_result:
                    self._process_single_match(match_result, spl_room, s_rooms, match_results, unmatched_results)
                    
            match_progress.progress(1.0)
            match_status.text(f"✅ 房型匹配完成，共匹配 {len(match_results)} 个房型，未匹配 {len(unmatched_results)} 个房型")
            
            # 处理匹配结果
            if match_results:
                # 使用process_matches处理并展示匹配结果
                match_df = self.process_matches(match_results, spl_rooms, s_rooms, model)
                
                # 显示未匹配房型
                if unmatched_results:
                    st.subheader("⚠️ 未匹配房型")
                    unmatched_df = pd.DataFrame(unmatched_results)
                    st.dataframe(unmatched_df, use_container_width=True)
                    
                    # 添加导出功能
                    st.download_button(
                        label="📥 导出未匹配数据",
                        data=unmatched_df.to_csv(index=False).encode('utf-8'),
                        file_name=f'未匹配数据_{spl_rooms[0].get("HotelID", "unknown")}.csv',
                        mime='text/csv',
                        key="download_unmatched"
                    )
                else:
                    st.info("所有房型均已匹配")
                
                # 显示所有匹配结果的导出功能
                if match_df is not None:
                    st.download_button(
                        label="📥 导出所有匹配结果",
                        data=match_df.to_csv(index=False).encode('utf-8'),
                        file_name=f'全部匹配结果_{spl_rooms[0].get("HotelID", "unknown")}.csv',
                        mime='text/csv',
                        key="download_all_matches"
                    )
            else:
                st.warning("没有找到符合阈值的匹配结果")
                
                # 显示所有未匹配房型
                if unmatched_results:
                    st.subheader("⚠️ 未匹配房型")
                    unmatched_df = pd.DataFrame(unmatched_results)
                    st.dataframe(unmatched_df, use_container_width=True)
                    
                    # 添加导出功能
                    st.download_button(
                        label="📥 导出未匹配数据",
                        data=unmatched_df.to_csv(index=False).encode('utf-8'),
                        file_name=f'未匹配数据_{spl_rooms[0].get("HotelID", "unknown")}.csv',
                        mime='text/csv',
                        key="download_unmatched_all"
                    )
        except Exception as e:
            st.error(f"匹配过程中发生错误: {str(e)}")
            self.logger.error(f"匹配过程中发生错误: {str(e)}", exc_info=True)

def get_available_models():
    """
    获取环境中所有可用的模型列表
    返回：包含模型名称和路径的字典
    """
    available_models = {}
    
    # 检查默认模型
    if os.path.exists(MODEL_PATH):
        available_models["默认模型"] = MODEL_PATH
    
    # 检查match_by_simcse目录下的所有checkpoints目录
    checkpoints_dir = os.path.join(BASE_DIR, "match_by_simcse")
    if os.path.exists(checkpoints_dir):
        for item in os.listdir(checkpoints_dir):
            if item.startswith("checkpoints_"):
                full_path = os.path.join(checkpoints_dir, item)
                if os.path.isdir(full_path):
                    # 检查是否有best_model子目录
                    best_model_path = os.path.join(full_path, "best_model")
                    if os.path.exists(best_model_path):
                        available_models[item] = best_model_path
                    else:
                        # 没有best_model子目录，使用目录本身
                        available_models[item] = full_path
    
    return available_models

def get_model_display_name(model_path: str) -> str:
    """
    根据模型路径获取一个友好的显示名称
    """
    # 检查是否是预设的模型名称
    for key, value in DEFAULT_VALUES["model_select"].items():
        if key in model_path:
            return f"{value} ({os.path.basename(model_path)})"
    
    # 如果不是预设模型，提取有用信息
    if "checkpoints_" in model_path:
        # 从路径中提取模型名，比如 checkpoints_512batch_5e
        model_name = model_path.split("checkpoints_")[-1].split("/")[0]
        return f"模型 {model_name}"
    
    # 如果是绝对路径，使用最后一个目录名
    return os.path.basename(os.path.dirname(model_path) if model_path.endswith("best_model") else model_path)

def main():
    """主函数"""
    try:
        logger.info("启动房型匹配分析工具")
        setup_debugger()
        viewer = RoomTypeViewer()
        viewer.run()
    except Exception as e:
        logger.error("程序运行出错", exc_info=True)
        st.error(f"程序运行出错: {str(e)}")

if __name__ == "__main__":
    main()

# 添加页面说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 🏨 房型匹配分析工具
    
    #### 功能特点
    - 支持单酒店详细分析和多国家批量处理
    - 实时展示处理进度和匹配结果
    - 灵活的数据筛选和导出功能
    - 直观的数据可视化展示
    
    #### 使用方法
    
    1. **选择处理模式**
       - 单酒店模式：深入分析单个酒店的房型匹配情况
       - 批量处理模式：批量分析指定国家的多个酒店
    
    2. **设置基本参数**
       - 选择国家：指定要分析的酒店所在国家
       - 选择酒店：可通过ID或名称搜索（单酒店模式）
       - 设置最大处理数量：限制批量处理的酒店数量
       - 选择匹配模型：选择要使用的房型匹配模型
       - 调整匹配阈值：设置房型匹配的精确度要求
    
    3. **查看分析结果**
       - 统计信息：房型数量、匹配状态分布等
       - 匹配详情：新增匹配、匹配变化等
       - 数据导出：支持导出各类分析结果
    
    #### 注意事项
    - 批量处理大量酒店时，建议先设置较小的处理数量进行测试
    - 可以通过调整匹配阈值来平衡匹配的准确度和覆盖率
    - 所有分析结果都支持导出，便于后续处理和分析
    """)

# 添加页脚
st.markdown("---")
st.markdown("#### 🏨 房型匹配分析工具 v2.0")
st.markdown("📊 支持单酒店分析和批量国家分析 | 💡 内置智能匹配算法 | 📈 实时数据可视化") 