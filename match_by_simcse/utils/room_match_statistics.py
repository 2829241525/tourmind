"""
房型匹配数据统计工具
用于统计各个酒店的房型匹配情况并生成报告
"""

import os
import sys
import pandas as pd
import requests
import logging
from typing import Dict, Optional, List
import warnings
from urllib3.exceptions import InsecureRequestWarning
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# 添加基础路径常量
BASE_DIR = "/home/maxon/disk2/roomMatch/room_match/"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('room_match_statistics.log')
    ]
)
logger = logging.getLogger(__name__)

# 忽略SSL警告
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

class RoomTypeAPI:
    """房型API接口类"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.logger = logging.getLogger(f"{__name__}.RoomTypeAPI")
        self.session = requests.Session()
    
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
            response = self.session.get(
                f"{self.base_url}/roomtypemapping/api/get_roomtype",
                params=params,
                headers=headers,
                verify=False,
                timeout=300  # 添加超时设置
            )
            
            if response.status_code != 200:
                self.logger.error(f"API请求失败 (hotel_id={hotel_id}): HTTP {response.status_code}")
                return None
                
            try:
                data = response.json()
                if not data or "Data" not in data:
                    self.logger.error(f"API返回数据格式错误 (hotel_id={hotel_id})")
                    return None
                return data
            except ValueError:
                self.logger.error(f"API返回数据解析失败 (hotel_id={hotel_id})")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.error(f"API请求超时 (hotel_id={hotel_id})")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API请求异常 (hotel_id={hotel_id}): {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"获取数据时发生未知错误 (hotel_id={hotel_id}): {str(e)}")
            return None

class RoomMatchStatistics:
    """房型匹配统计类"""
    
    def __init__(self, max_workers=10, limit=100):
        self.api = RoomTypeAPI("http://erp.tourmind.cn")
        self.logger = logging.getLogger(f"{__name__}.RoomMatchStatistics")
        self.max_workers = max_workers
        self.limit = limit
        
        # API调用的默认参数
        self.default_params = {
            "supplier_id": "0",
            "room_base_type": "tourmind",
            "room_status": "0",
            "cookie": "admintools_user_session=MTc0MTE0NTkwN3xiTWRlWWJWbHAydThyclhuQVczOExtTW9FMzkyWEJ5T0lhb1h1MXlnVEIyQ2xmZWVfZFF2cFdQMVRJMnNmYm94MlR5X2JxcGZkQXM9fDMKwBADbxvlPFMkmacoVAR9ZrkTkbgR0CXwkCYcpWzq",
            "operator": "xiemingxuan",
            "referer": "http://erp.tourmind.cn/roomtypemapping/961935?supplierId=0&brt=tourmind&loadMaster=true"
        }
    
    def load_hotel_data(self) -> pd.DataFrame:
        """加载酒店数据"""
        try:
            file_path = os.path.join(BASE_DIR, "match_by_simcse/data/匹配测试酒店.csv")
            
            if not os.path.exists(file_path):
                self.logger.error(f"酒店数据文件不存在: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(file_path)
            
            # 数据预处理
            df['hotel_id'] = df['hotel_id'].astype(str)
            df['country_code'] = df['country_code'].fillna('Unknown')
            df['country_code'] = df['country_code'].astype(str)
            df['hotel_name'] = df['hotel_name'].fillna('未知酒店名称')
            df['hotel_name'] = df['hotel_name'].astype(str)
            
            # 限制处理数量
            if self.limit > 0:
                df = df.head(self.limit)
            
            return df
        except Exception as e:
            self.logger.error(f"加载酒店数据文件失败: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def process_room_data(self, data: Dict) -> Dict:
        """处理房型数据，返回统计结果"""
        try:
            if data is None:
                return {
                    "total_rooms": 0,
                    "matched_rooms": 0,
                    "unmatched_rooms": 0,
                    "match_rate": 0,
                    "supplier_counts": {},
                    "standard_rooms": 0
                }
                
            hotel_data = data.get("Data")
            if hotel_data is None:
                return {
                    "total_rooms": 0,
                    "matched_rooms": 0,
                    "unmatched_rooms": 0,
                    "match_rate": 0,
                    "supplier_counts": {},
                    "standard_rooms": 0
                }
                
            spl_rooms = hotel_data.get("SPLRoomTypeList", [])
            s_rooms = hotel_data.get("SRoomTypeList", [])
            
            # 统计匹配情况
            total_spl_rooms = len(spl_rooms)
            matched_rooms = sum(1 for room in spl_rooms if room.get("SRoomID", 0) > 0)
            unmatched_rooms = total_spl_rooms - matched_rooms
            match_rate = (matched_rooms / total_spl_rooms * 100) if total_spl_rooms > 0 else 0
            
            # 统计每个供应商的房型数量
            supplier_counts = {}
            for room in spl_rooms:
                supplier_id = str(room.get("SupplierID", "unknown"))
                supplier_counts[supplier_id] = supplier_counts.get(supplier_id, 0) + 1
            
            return {
                "total_rooms": total_spl_rooms,
                "matched_rooms": matched_rooms,
                "unmatched_rooms": unmatched_rooms,
                "match_rate": match_rate,
                "supplier_counts": supplier_counts,
                "standard_rooms": len(s_rooms)
            }
        except Exception as e:
            self.logger.error(f"处理房型数据时发生错误: {str(e)}", exc_info=True)
            return {
                "total_rooms": 0,
                "matched_rooms": 0,
                "unmatched_rooms": 0,
                "match_rate": 0,
                "supplier_counts": {},
                "standard_rooms": 0
            }
    
    def process_single_hotel(self, row) -> Dict:
        """处理单个酒店的数据"""
        hotel_id = row['hotel_id']
        
        # 获取房型数据
        room_data = self.api.fetch_room_data(
            hotel_id=hotel_id,
            supplier_id=self.default_params["supplier_id"],
            room_base_type=self.default_params["room_base_type"],
            room_status=self.default_params["room_status"],
            cookie=self.default_params["cookie"],
            operator=self.default_params["operator"],
            referer=self.default_params["referer"]
        )
        
        if room_data:
            stats = self.process_room_data(room_data)
            
            # 准备统计数据，只包含主要信息
            hotel_stats = {
                "酒店ID": hotel_id,
                "酒店名称": row['hotel_name'],
                "国家": row['country_code'],
                "总房型数": stats["total_rooms"],
                "规则匹配数": stats["matched_rooms"],
                "未匹配数": stats["unmatched_rooms"],
                "规则匹配率": f"{stats['match_rate']:.2f}%",
                "标准房型数": stats["standard_rooms"],
                "供应商数量": len(stats["supplier_counts"])
            }
            
            return hotel_stats
        else:
            return {
                "酒店ID": hotel_id,
                "酒店名称": row['hotel_name'],
                "国家": row['country_code'],
                "总房型数": 0,
                "规则匹配数": 0,
                "未匹配数": 0,
                "规则匹配率": "0.00%",
                "标准房型数": 0,
                "供应商数量": 0
            }
    
    def generate_statistics(self) -> pd.DataFrame:
        """生成统计报告"""
        hotels_df = self.load_hotel_data()
        if hotels_df.empty:
            self.logger.error("无法加载酒店数据")
            return pd.DataFrame()
        
        statistics_data = []
        total_hotels = len(hotels_df)
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_row = {executor.submit(self.process_single_hotel, row): row for _, row in hotels_df.iterrows()}
            
            # 使用tqdm显示进度条
            with tqdm(total=total_hotels, desc="处理进度") as pbar:
                for future in as_completed(future_to_row):
                    try:
                        hotel_stats = future.result()
                        statistics_data.append(hotel_stats)
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(f"处理酒店时发生错误: {str(e)}")
        
        # 转换为DataFrame
        stats_df = pd.DataFrame(statistics_data)
        
        # 添加汇总行，修改规则匹配率的计算
        total_matched = stats_df["规则匹配数"].sum()
        total_rooms = stats_df["总房型数"].sum()
        overall_match_rate = (total_matched / total_rooms * 100) if total_rooms > 0 else 0
        
        summary = {
            "酒店ID": "汇总",
            "酒店名称": f"共{len(stats_df)}家酒店",
            "国家": "ALL",
            "总房型数": total_rooms,
            "规则匹配数": total_matched,
            "未匹配数": stats_df["未匹配数"].sum(),
            "规则匹配率": f"{overall_match_rate:.2f}%",
            "标准房型数": stats_df["标准房型数"].max(),
            "供应商数量": stats_df["供应商数量"].mean()
        }
        
        stats_df = pd.concat([stats_df, pd.DataFrame([summary])], ignore_index=True)
        
        # 设置显示的列顺序
        columns_order = [
            "酒店ID", "酒店名称", "国家", "总房型数", "规则匹配数", 
            "未匹配数", "规则匹配率", "标准房型数", "供应商数量"
        ]
        stats_df = stats_df[columns_order]
        
        # 设置pandas显示选项
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        
        return stats_df

def main():
    """主函数"""
    try:
        logger.info("开始生成房型匹配统计报告")
        
        # 创建统计实例，设置并发数和处理限制
        stats = RoomMatchStatistics(max_workers=10, limit=100)
        
        # 生成统计报告
        stats_df = stats.generate_statistics()
        
        if not stats_df.empty:
            # 保存报告
            output_file = "房型匹配统计报告.csv"
            stats_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"统计报告已保存到: {output_file}")
            
            # 打印汇总信息
            summary = stats_df.iloc[-1]
            logger.info("\n=== 统计汇总 ===")
            logger.info(f"总酒店数: {len(stats_df)-1}")
            logger.info(f"总房型数: {summary['总房型数']}")
            logger.info(f"规则匹配数: {summary['规则匹配数']}")
            logger.info(f"未匹配数: {summary['未匹配数']}")
            logger.info(f"规则匹配率: {summary['规则匹配率']}")
            logger.info(f"平均供应商数量: {summary['供应商数量']:.1f}")
            
            # 显示前10行数据
            logger.info("\n=== 数据预览（前10行）===")
            print(stats_df.head(10).to_string())
        else:
            logger.error("生成统计报告失败")
    
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 