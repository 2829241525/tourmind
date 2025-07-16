#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from hotel_room_extractor import HotelRoomExtractor


def main():
    # 初始化提取器
    extractor = HotelRoomExtractor(max_workers=4)

    # 处理酒店
    output_file = extractor.process_hotels_from_csv_multithreaded(
        csv_file_path="1000sampled_hotels.csv",
        output_dir="output",
        start_index=0,
        end_index=20,
        delay_seconds=1.0
    )

    print(f"处理完成，输出文件: {output_file}")


if __name__ == "__main__":
    main()
