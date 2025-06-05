#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json
import random
import argparse
from typing import List, Dict, Any, Optional

# 配置信息
config = {
    'api_base_url': 'http://0.0.0.0:8000',  # 服务地址
    'sample_size': 5,  # 默认样本大小
    'n_clusters': None,  # 默认聚类数量，None表示自动计算
    'min_cluster_size': 2,  # 最小聚类大小
    'timeout': 10,  # 请求超时时间（秒）
}

# 房型文本样例（用于测试）
SAMPLE_DESCRIPTIONS = [
    "标准大床房，40平米，1张1.8米大床，有窗，可住2人",
    "豪华大床房，50平米，1张2米大床，有窗，可住2人",
    "行政大床房，60平米，1张2米大床，有窗，可住2人，含行政礼遇",
    "标准双床房，45平米，2张1.2米单人床，有窗，可住2人",
    "豪华双床房，55平米，2张1.35米单人床，有窗，可住2人",
    "行政双床房，65平米，2张1.35米单人床，有窗，可住2人，含行政礼遇",
    "家庭房，70平米，1张1.8米大床和2张1.2米单人床，有窗，可住4人",
    "套房，80平米，1张2米大床，独立客厅，有窗，可住2人",
    "总统套房，120平米，1张2米大床，独立客厅和餐厅，有窗，可住2人",
    "商务套房，100平米，1张2米大床，独立办公区和会客区，有窗，可住2人"
]


def get_random_rooms(count: int = 5) -> List[Dict[str, str]]:
    """生成随机房间数据用于测试"""
    rooms = []
    for i in range(count):
        room_id = f"room_{random.randint(1000, 9999)}"
        description = random.choice(SAMPLE_DESCRIPTIONS)
        rooms.append({
            "room_id": room_id,
            "description": description
        })
    return rooms


def group_rooms(
    rooms: List[Dict[str, str]],
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 2
) -> Dict[str, Any]:
    """调用房型分组API"""
    url = f"{config['api_base_url']}/group"

    payload = {
        "rooms": rooms,
        "n_clusters": n_clusters,
        "min_cluster_size": min_cluster_size
    }

    print(f"正在发送请求到: {url}")
    print(f"请求参数: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=config['timeout']
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP错误: {e}")
        print(f"响应内容: {e.response.text}")
        return {"error": str(e)}
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return {"error": str(e)}


def get_config() -> Dict[str, Any]:
    """获取服务配置信息"""
    url = f"{config['api_base_url']}/config"

    print(f"正在获取配置信息: {url}")

    try:
        response = requests.get(url, timeout=config['timeout'])
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"获取配置失败: {e}")
        return {"error": str(e)}


def get_health() -> Dict[str, Any]:
    """获取服务健康状态"""
    url = f"{config['api_base_url']}/health"

    print(f"正在检查服务健康状态: {url}")

    try:
        response = requests.get(url, timeout=config['timeout'])
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"健康检查失败: {e}")
        return {"error": str(e)}


def get_stats() -> Dict[str, Any]:
    """获取服务统计信息"""
    url = f"{config['api_base_url']}/stats"

    print(f"正在获取服务统计信息: {url}")

    try:
        response = requests.get(url, timeout=config['timeout'])
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"获取统计信息失败: {e}")
        return {"error": str(e)}


def print_curl_commands():
    """打印curl命令用于测试"""
    print("\n=== 以下是访问各个接口的curl命令 ===")

    # 分组接口curl命令
    group_curl = f"""curl -X POST "{config['api_base_url']}/group" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "rooms": [
      {{"room_id": "room_1001", "description": "标准大床房，40平米，1张1.8米大床，有窗，可住2人"}},
      {{"room_id": "room_1002", "description": "豪华大床房，50平米，1张2米大床，有窗，可住2人"}},
      {{"room_id": "room_1003", "description": "行政大床房，60平米，1张2米大床，有窗，可住2人，含行政礼遇"}},
      {{"room_id": "room_1004", "description": "标准双床房，45平米，2张1.2米单人床，有窗，可住2人"}},
      {{"room_id": "room_1005", "description": "豪华双床房，55平米，2张1.35米单人床，有窗，可住2人"}}
    ],
    "n_clusters": null,
    "min_cluster_size": 2
  }}'"""

    # 配置接口curl命令
    config_curl = f"""curl -X GET "{config['api_base_url']}/config" """

    # 健康检查接口curl命令
    health_curl = f"""curl -X GET "{config['api_base_url']}/health" """

    # 统计接口curl命令
    stats_curl = f"""curl -X GET "{config['api_base_url']}/stats" """

    print("\n=== 分组接口 ===")
    print(group_curl)
    print("\n=== 配置接口 ===")
    print(config_curl)
    print("\n=== 健康检查接口 ===")
    print(health_curl)
    print("\n=== 统计接口 ===")
    print(stats_curl)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="房型分组服务客户端")
    parser.add_argument("--action", choices=["group", "config", "health", "stats", "curl"],
                        default="group", help="要执行的操作")
    parser.add_argument("--sample-size", type=int, default=config['sample_size'],
                        help="随机生成的房型数量")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="聚类数量，为None时自动计算")
    parser.add_argument("--min-cluster-size", type=int, default=config['min_cluster_size'],
                        help="最小聚类大小")
    parser.add_argument("--api-base-url", default=config['api_base_url'],
                        help="API基础URL")

    args = parser.parse_args()

    # 更新配置
    config['api_base_url'] = args.api_base_url
    config['sample_size'] = args.sample_size
    config['n_clusters'] = args.n_clusters
    config['min_cluster_size'] = args.min_cluster_size

    if args.action == "group":
        # 生成随机房型数据
        rooms = get_random_rooms(count=config['sample_size'])
        # 调用分组API
        result = group_rooms(
            rooms=rooms,
            n_clusters=config['n_clusters'],
            min_cluster_size=config['min_cluster_size']
        )
        print("\n=== 分组结果 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.action == "config":
        result = get_config()
        print("\n=== 服务配置 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.action == "health":
        result = get_health()
        print("\n=== 健康状态 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.action == "stats":
        result = get_stats()
        print("\n=== 服务统计 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.action == "curl":
        print_curl_commands()


if __name__ == "__main__":
    main()
