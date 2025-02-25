import json
import glob
import os
from datetime import datetime

def check_cache_files():
    cache_dir = 'room_match/data/cache_supply'
    cache_files = glob.glob(os.path.join(cache_dir, 'api_cache_*.json'))
    
    # 收集所有键和数据
    all_keys = set()
    duplicate_keys = set()
    key_info = {}  # 存储每个key的信息：{key: [(file, update_time), ...]}
    total_entries = 0
    
    print(f'开始检查 {len(cache_files)} 个缓存文件...\n')
    
    for cache_file in sorted(cache_files):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                file_size = os.path.getsize(cache_file)
                keys = set(data.keys())
                total_entries += len(keys)
                
                # 检查每个key的更新时间
                for key in keys:
                    update_time = datetime.fromisoformat(data[key]['update_time'])
                    if key not in key_info:
                        key_info[key] = []
                    key_info[key].append((os.path.basename(cache_file), update_time))
                
                # 检查重复
                duplicates = keys & all_keys
                if duplicates:
                    duplicate_keys.update(duplicates)
                
                all_keys.update(keys)
                
                print(f'文件: {os.path.basename(cache_file)}')
                print(f'大小: {file_size / (1024*1024):.2f}MB')
                print(f'条目数: {len(keys)}')
                print('---')
        except Exception as e:
            print(f'读取文件 {cache_file} 时出错: {str(e)}')
    
    print('\n重复数据分析:')
    duplicate_with_diff_time = 0
    for key in duplicate_keys:
        if len(key_info[key]) > 1:
            # 检查是否有不同的更新时间
            times = [info[1] for info in key_info[key]]
            if len(set(times)) > 1:
                duplicate_with_diff_time += 1
                print(f'\n键 {key} 在多个文件中有不同的更新时间:')
                for file_name, update_time in sorted(key_info[key], key=lambda x: x[1]):
                    print(f'  {file_name}: {update_time}')
    
    print('\n总结:')
    print(f'总条目数: {total_entries}')
    print(f'唯一键数量: {len(all_keys)}')
    print(f'重复键数量: {len(duplicate_keys)}')
    print(f'具有不同更新时间的重复键数量: {duplicate_with_diff_time}')
    if duplicate_keys:
        print(f'重复率: {(len(duplicate_keys) / total_entries * 100):.2f}%')

if __name__ == '__main__':
    check_cache_files() 