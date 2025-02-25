import psutil
import time
import os
from datetime import datetime
from typing import Union
import sys

def get_size(bytes: int) -> str:
    """
    将字节转换为人类可读的格式
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}"
        bytes /= 1024

def create_progress_bar(percentage: float, width: int = 30) -> str:
    """创建进度条"""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:.1f}%"

def format_usage(used: Union[float, int], total: Union[float, int], width: int = 30) -> str:
    """格式化使用率显示"""
    percentage = (used / total) * 100
    return create_progress_bar(percentage, width)

def move_cursor_up(lines: int):
    """将光标向上移动n行"""
    sys.stdout.write(f'\033[{lines}A')
    sys.stdout.flush()

def clear_lines(lines: int):
    """清除从当前行开始的n行"""
    for _ in range(lines):
        sys.stdout.write('\033[2K\033[1G')  # 清除当前行并回到行首
        sys.stdout.write('\033[1B')  # 下移一行
    move_cursor_up(lines)  # 回到起始位置

def get_cpu_info():
    """获取CPU信息"""
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    return (f"CPU使用率: {create_progress_bar(cpu_percent)}\n"
            f"CPU核心数: {cpu_count}\n"
            f"CPU频率: {cpu_freq.current:.1f}MHz")

def get_memory_info():
    """获取内存信息"""
    memory = psutil.virtual_memory()
    return (f"内存使用情况:\n"
            f"总内存: {get_size(memory.total)}\n"
            f"使用率: {format_usage(memory.used, memory.total)}\n"
            f"已用: {get_size(memory.used)} | 可用: {get_size(memory.available)}")

def get_network_info():
    """获取网络信息"""
    net_io = psutil.net_io_counters()
    return (f"网络流量:\n"
            f"↑ 发送: {get_size(net_io.bytes_sent)}\n"
            f"↓ 接收: {get_size(net_io.bytes_recv)}\n"
            f"发送包数: {net_io.packets_sent:,}\n"
            f"接收包数: {net_io.packets_recv:,}")

def count_display_lines(display_text: str) -> int:
    """计算显示内容的行数"""
    return len(display_text.split('\n'))

def monitor_system():
    """系统监控主函数"""
    try:
        previous_net_io = psutil.net_io_counters()
        first_run = True
        previous_lines = 0
        
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 计算网络速率
            current_net_io = psutil.net_io_counters()
            send_speed = (current_net_io.bytes_sent - previous_net_io.bytes_sent)
            recv_speed = (current_net_io.bytes_recv - previous_net_io.bytes_recv)
            previous_net_io = current_net_io
            
            # 组装显示信息
            display = [
                f"系统监控 - {current_time}",
                f"{'='*50}",
                get_cpu_info(),
                f"{'='*50}",
                get_memory_info(),
                f"{'='*50}",
                get_network_info(),
                f"当前网速:",
                f"↑ 上传: {get_size(send_speed)}/s",
                f"↓ 下载: {get_size(recv_speed)}/s",
                f"{'='*50}",
                "按 Ctrl+C 退出监控"
            ]
            
            display_text = '\n'.join(display)
            current_lines = count_display_lines(display_text)
            
            if not first_run:
                # 清除之前的输出
                clear_lines(previous_lines)
            else:
                first_run = False
            
            # 打印新的信息
            print(display_text)
            previous_lines = current_lines
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    # 检查是否支持ANSI转义序列
    if os.name == 'nt':  # Windows系统
        try:
            from ctypes import windll
            k = windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except:
            print("警告: Windows系统可能不支持ANSI转义序列，显示效果可能不正常")
    
    monitor_system() 