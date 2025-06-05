#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import nacos
import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Any, Optional
from tqdm import tqdm

def get_nacos_config() -> Dict[str, str]:
    """
    从Nacos获取MySQL配置信息
    """
    try:
        # 从环境变量获取Nacos配置
        nacos_addr = os.getenv('NACOS_ADDR')
        nacos_ns = os.getenv('NACOS_NS')
        
        if not nacos_addr or not nacos_ns:
            raise ValueError("环境变量 NACOS_ADDR 或 NACOS_NS 未设置")

        # 创建Nacos客户端
        client = nacos.NacosClient(nacos_addr, namespace=nacos_ns)
        
        # 获取配置
        config = client.get_config(
            data_id='cn.tourmind.hlz.mysql.toumind',
            group='hlz'
        )
        
        if not config:
            raise ValueError("无法从Nacos获取配置信息")
            
        # 解析JSON配置
        config_dict = json.loads(config)
        required_keys = ['host', 'user', 'password']
        
        if not all(key in config_dict for key in required_keys):
            raise ValueError("Nacos配置缺少必要的键值")
            
        print("成功从Nacos获取配置信息")
        return config_dict
        
    except Exception as e:
        print(f"获取Nacos配置失败: {str(e)}")
        raise

def create_db_connection(config: Dict[str, str]) -> mysql.connector.MySQLConnection:
    """
    创建MySQL数据库连接
    """
    try:
        # 解析主机和端口
        host_parts = config['host'].split(':')
        host = host_parts[0]
        port = int(host_parts[1]) if len(host_parts) > 1 else 3306
        
        # 创建连接
        connection = mysql.connector.connect(
            host=host,
            port=port,
            user=config['user'],
            password=config['password'],
            database='toumind'
        )
        
        print("成功连接到MySQL数据库")
        return connection
        
    except Error as e:
        print(f"数据库连接失败: {str(e)}")
        raise

def extract_room_names(connection: mysql.connector.MySQLConnection) -> List[Dict[str, str]]:
    """
    从数据库中提取房型名称和房型中文名称
    """
    room_data_list = []
    last_room_id = 0
    batch_size = 10000
    
    try:
        cursor = connection.cursor()
        
        # Get total count for progress bar
        cursor.execute("SELECT COUNT(*) FROM s_room")
        total_rooms = cursor.fetchone()[0]
        
        with tqdm(total=total_rooms, unit="room", desc="Extracting room names") as pbar:
            while True:
                # 执行批量查询
                query = """
                    SELECT s_room_id, room_name, room_name_cn 
                    FROM s_room 
                    WHERE s_room_id > %s 
                    ORDER BY s_room_id ASC 
                    LIMIT %s
                """
                cursor.execute(query, (last_room_id, batch_size))
                results = cursor.fetchall()
                
                if not results:
                    break
                    
                # 处理结果
                for room_id, room_name, room_name_cn in results:
                    room_data_list.append({"room_name": room_name, "room_name_cn": room_name_cn})
                    last_room_id = room_id
                
                pbar.update(len(results))
            
        cursor.close()
        return room_data_list
        
    except Error as e:
        print(f"数据提取过程中发生错误: {str(e)}")
        raise

def save_raw_room_data(room_data_list: List[Dict[str, str]], output_file: str = "raw_room_data.json"):
    """
    将提取的原始房型数据保存到JSON文件
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(room_data_list, f, ensure_ascii=False, indent=2)
            
        print(f"原始房型数据已保存到 {output_file}")
        
    except Exception as e:
        print(f"保存原始数据时发生错误: {str(e)}")
        raise

def main():
    try:
        # 获取Nacos配置
        config = get_nacos_config()
        
        # 创建数据库连接
        connection = create_db_connection(config)
        
        try:
            # 提取房型名称和中文名称
            print("开始从数据库提取房型数据...")
            room_data_list = extract_room_names(connection)
            print(f"数据提取完成，共提取 {len(room_data_list)} 条房型数据")
            
            # 保存原始数据
            save_raw_room_data(room_data_list) # Default output file "raw_room_data.json"
            
        finally:
            # 确保关闭数据库连接
            if connection.is_connected():
                connection.close()
                print("数据库连接已关闭")
                
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 