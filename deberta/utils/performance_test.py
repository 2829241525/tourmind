import requests
import concurrent.futures
import time
import json
import statistics
from datetime import datetime
import os
from pathlib import Path
import threading
from queue import Queue

# 配置
CONFIG = {
    'url': 'http://121.15.198.184:13001/match/',
    'num_threads': 8,
    'test_duration': 300,  # 测试持续时间（秒）
    'timeout': 30,  # 请求超时时间（秒）
    'log_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'perf_logs'),
    'progress_interval': 30,  # 进度报告间隔（秒）
}

# 测试数据
TEST_DATA = {
    "spl_room_names": [
        "Deluxe Double room (full double bed) (king size bed, bath)",
        "Habitación Estándar",
        "Villa (Pool Villa)",
        "Suite, 1 Bedroom, Non Smoking",
        "Double or Twin Room (Cycling Package)",
        "Deluxe Room, 2 Queen Beds"
    ],
    "spl_room_bed_names": [
        "double king size bed",
        "",
        "",
        "1 King Bed and 1 Queen Bed",
        "2 Twin Beds/1 Double Bed",
        ""
    ],
    "s_room_names": [
        "Deluxe Room, 1 King Bed",
        "Standard Room",
        "Room (Garden Villa)",
        "Suite, 1 Bedroom, Non Smoking",
        "Double or Twin Room (Cycling Package)",
        "Deluxe Room, 2 Double Beds"
    ],
    "s_room_bed_names": [
        "1 King Bed",
        "",
        "",
        "1 King Bed",
        "1 Double Bed/2 Twin Beds",
        ""
    ],
    "token": "tourmind_room_match"
}

# 请求头
HEADERS = {
    'Content-Type': 'application/json',
    'Accept': '*/*',
    'Host': '10.0.0.110:13001',
    'Connection': 'keep-alive',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
}


def make_request():
    """执行单个请求并返回响应时间和状态"""
    start_time = time.time()
    try:
        response = requests.post(
            CONFIG['url'],
            headers=HEADERS,
            json=TEST_DATA,
            timeout=CONFIG['timeout']
        )
        elapsed_time = time.time() - start_time
        return {
            'success': response.status_code == 200,
            'status_code': response.status_code,
            'elapsed_time': elapsed_time,
            'response_length': len(response.text) if response.status_code == 200 else 0,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'success': False,
            'status_code': -1,
            'elapsed_time': elapsed_time,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def worker(worker_id, results_queue, stop_event):
    """工作线程函数"""
    while not stop_event.is_set():
        result = make_request()
        result['worker_id'] = worker_id
        results_queue.put(result)


def print_progress(start_time, results):
    """打印当前进度和性能指标"""
    current_time = time.time()
    elapsed = current_time - start_time
    successful_requests = [r for r in results if r['success']]
    if not successful_requests:
        return

    response_times = [r['elapsed_time'] for r in successful_requests]
    current_qps = len(successful_requests) / elapsed

    print(f"\n当前测试进度 ({elapsed:.0f}秒/{CONFIG['test_duration']}秒):")
    print(f"已完成请求数: {len(results)}")
    print(f"成功请求数: {len(successful_requests)}")
    print(f"当前 QPS: {current_qps:.2f}")
    print(f"平均响应时间: {statistics.mean(response_times)*1000:.2f} ms")
    print(f"最近最大响应时间: {max(response_times)*1000:.2f} ms")


def run_performance_test():
    """运行性能测试"""
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(CONFIG['log_dir'], f'perf_test_{timestamp}.json')

    print(f"开始性能测试 (持续时间: {CONFIG['test_duration']}秒)...")
    start_time = time.time()
    results_queue = Queue()
    stop_event = threading.Event()
    all_results = []

    # 创建并启动工作线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['num_threads']) as executor:
        futures = [
            executor.submit(worker, i, results_queue, stop_event)
            for i in range(CONFIG['num_threads'])
        ]

        # 收集结果直到达到指定时间
        last_progress = start_time
        try:
            while time.time() - start_time < CONFIG['test_duration']:
                try:
                    result = results_queue.get(timeout=1)
                    all_results.append(result)

                    # 定期打印进度
                    current_time = time.time()
                    if current_time - last_progress >= CONFIG['progress_interval']:
                        print_progress(start_time, all_results)
                        last_progress = current_time

                except Queue.Empty:
                    continue
        finally:
            stop_event.set()

        # 等待所有线程完成
        for future in futures:
            future.result()

    total_time = time.time() - start_time

    # 计算最终统计信息
    successful_requests = [r for r in all_results if r['success']]
    failed_requests = [r for r in all_results if not r['success']]
    response_times = [r['elapsed_time'] for r in successful_requests]

    if response_times:
        stats = {
            'total_requests': len(all_results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'total_time': total_time,
            'avg_response_time': statistics.mean(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'median_response_time': statistics.median(response_times),
            # 95th percentile
            'p95_response_time': statistics.quantiles(response_times, n=20)[18],
            # 99th percentile
            'p99_response_time': statistics.quantiles(response_times, n=100)[98],
            'requests_per_second': len(successful_requests) / total_time
        }
    else:
        stats = {
            'total_requests': len(all_results),
            'successful_requests': 0,
            'failed_requests': len(failed_requests),
            'total_time': total_time,
            'requests_per_second': 0
        }

    # 保存详细结果到文件
    test_results = {
        'config': CONFIG,
        'statistics': stats,
        'detailed_results': all_results
    }

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    # 打印最终摘要
    print("\n性能测试最终结果:")
    print(f"总请求数: {stats['total_requests']}")
    print(f"成功请求数: {stats['successful_requests']}")
    print(f"失败请求数: {stats['failed_requests']}")
    print(f"总执行时间: {stats['total_time']:.2f} 秒")

    if response_times:
        print(f"平均响应时间: {stats['avg_response_time']*1000:.2f} ms")
        print(f"最小响应时间: {stats['min_response_time']*1000:.2f} ms")
        print(f"最大响应时间: {stats['max_response_time']*1000:.2f} ms")
        print(f"中位数响应时间: {stats['median_response_time']*1000:.2f} ms")
        print(f"95分位响应时间: {stats['p95_response_time']*1000:.2f} ms")
        print(f"99分位响应时间: {stats['p99_response_time']*1000:.2f} ms")

    print(f"每秒请求数 (QPS): {stats['requests_per_second']:.2f}")
    print(f"\n详细结果已保存到: {log_file}")


if __name__ == '__main__':
    run_performance_test()
