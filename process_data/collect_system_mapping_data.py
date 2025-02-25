import http.client
import json
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import sys
import os

###########################
#根据日志接口，获取所有数据，分时间获取
###########################

def fetch_data_with_retry(start_date, end_date, page=1, page_size=10000, max_retries=3):
    """获取指定时间范围内的数据，带重试机制"""
    for attempt in range(max_retries):
        try:
            conn = http.client.HTTPConnection("119.23.73.59", 8080, timeout=30)
            payload = json.dumps({
                "password": "your-secure-password",
                "start_date": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_date": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "page": page,
                "page_size": page_size
            })
            headers = {
                'Content-Type': 'application/json'
            }

            conn.request("POST", "/api/room-log", payload, headers)
            res = conn.getresponse()
            data = res.read()
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\nError fetching data (attempt {attempt + 1}/{max_retries}): {e}")
                return None
            print(f"\n重试第 {attempt + 1} 次...")
            time.sleep(2 ** attempt)  # 指数退避
        finally:
            conn.close()


def get_cache_file_name(date):
    """获取缓存文件名，按月存储
    Args:
        date: datetime对象
    Returns:
        str: 缓存文件路径
    """
    cache_dir = "room_match/logs/collect_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return os.path.join(cache_dir, f"data_{date.strftime('%Y%m')}.csv")


def collect_data_for_year(start_date, end_date, temp_file="temp_data.csv", target_records=300000):
    """收集全年的system_mapping数据，直到达到目标数量"""
    total_records = 0
    current_date = start_date
    first_write = True

    # 创建进度条
    total_days = (end_date - start_date).days
    pbar = tqdm(total=total_days, desc=f"收集 {start_date.year} 年数据")

    try:
        while current_date < end_date and total_records < target_records:
            next_date = current_date + timedelta(days=1)
            
            # 从API获取数据
            page = 1
            all_data = []
            retries = 0
            max_retries = 3

            while retries < max_retries:
                try:
                    while True:
                        response = fetch_data_with_retry(current_date, next_date, page, page_size=10000)

                        if not response or not response.get("success"):
                            break

                        data = response.get("data", {})
                        items = data.get("data", [])

                        if not items:
                            break

                        filtered_data = [item for item in items if item.get("user_name") == "system_mapping"]

                        if filtered_data:
                            all_data.extend(filtered_data)

                        total_pages = data.get("total_pages", 0)
                        if page >= total_pages or len(all_data) >= 2000:
                            break

                        page += 1
                        time.sleep(0.01)

                    if all_data:
                        df = pd.DataFrame(all_data)
                        if len(df) > 2000:
                            df = df.sample(n=2000, random_state=42)
                        df['date'] = current_date.strftime("%Y-%m-%d")
                        daily_data = df.to_dict('records')
                    else:
                        daily_data = []
                    break

                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        print(f"\n获取数据失败: {current_date.date()}, {str(e)}")
                        break
                    print(f"\n重试获取数据: {current_date.date()}, 第{retries}次")
                    time.sleep(2 ** retries)

            # 将当天的数据保存到结果文件
            if daily_data:
                df = pd.DataFrame(daily_data)
                df['date'] = current_date.strftime("%Y-%m-%d")
                if first_write:
                    df.to_csv(temp_file, index=False)
                    first_write = False
                else:
                    df.to_csv(temp_file, mode='a', header=False, index=False)

                total_records += len(daily_data)
                print(f"\n获取数据: {current_date.date()} ({len(daily_data):,d}条)")

            current_date = next_date
            pbar.update(1)
            pbar.set_description(f"收集数据 (已收集: {total_records:,d}/{target_records:,d})")

            if total_records >= target_records:
                print(f"\n已达到目标数据量: {total_records:,d} 条")
                break
    finally:
        pbar.close()

    return total_records

def process_and_save_data(output_file="room_match/data/source_system_data.csv", temp_file="room_match/data/temp_data.csv"):
    """处理并保存数据到CSV文件"""
    try:
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2025, 1, 1)  # 获取2024全年数据
        target_records = 500000  # 目标50万条数据

        print(f"开始收集数据...")
        total_records = collect_data_for_year(start_date, end_date, temp_file, target_records)

        if total_records == 0:
            print("未收集到数据")
            return

        print(f"\n总共收集到 {total_records:,d} 条数据")

        # 读取并显示数据分布
        df = pd.read_csv(temp_file)
        daily_distribution = df['date'].value_counts().sort_index()
        print("\n每日数据分布:")
        print(daily_distribution)

        # 重命名最终文件
        if os.path.exists(temp_file):
            os.rename(temp_file, output_file)
            print(f"\n数据已保存到: {output_file}")

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n程序出错: {e}")
        raise


if __name__ == "__main__":
    process_and_save_data()
