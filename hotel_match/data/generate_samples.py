import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os
import psutil
import warnings
warnings.filterwarnings('ignore')

# 配置
config = {
    'input_file': '/home/maxon/disk2/roomMatch/room_match/hotel_match/data/merged_high_score.csv',
    'output_file': '/home/maxon/disk2/roomMatch/room_match/hotel_match/data/hotel_samples.csv',
    'random_seed': 42,
    'neg_pos_ratio': 4,  # 负样本:正样本 = 4:1
    'chunk_size': 1000,  # 分批处理的大小
    # 使用CPU核心数-1的进程数，但最多8个进程
    'n_processes': min(max(1, mp.cpu_count() - 1), 8),
    'batch_size': 50000,  # 批处理大小
    'lat_lng_threshold': 0.05,  # 经纬度差值粗筛阈值，大约5公里
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def init_worker():
    """初始化工作进程"""
    # 设置进程优先级
    p = psutil.Process()
    p.nice(10)  # 降低优先级，避免影响系统
    # 设置随机种子
    np.random.seed(os.getpid() % 2**32)


def create_text_format_vectorized(df, is_supply=True):
    """向量化创建文本格式"""
    if is_supply:
        # 处理城市名，如果为空则不添加
        city_name_part = df['CityName'].fillna('')
        city_name_mask = city_name_part != ''
        city_name_text = np.where(city_name_mask,
                                  ",CityName:" + city_name_part.astype(str),
                                  "")

        return (
            "HotelName:" + df['HotelName'].fillna('').astype(str) +
            city_name_text +
            ",Address:" + df['Address'].fillna('').astype(str)
        )
    else:
        # 处理目标城市名，如果为空则不添加
        target_city_name_part = df['MappingWorksCityName'].fillna('')
        target_city_name_mask = target_city_name_part != ''
        target_city_name_text = np.where(target_city_name_mask,
                                         ",TargetCityName:" +
                                         target_city_name_part.astype(str),
                                         "")

        return (
            "TargetHotel:" + df['MappingWorksHotelName'].fillna('').astype(str) +
            target_city_name_text +
            ",TargetAddress:" +
            df['MappingWorksAddress'].fillna('').astype(str)
        )


def calculate_distance(lat1, lng1, lat2, lng2):
    """计算两点之间的欧几里得距离（快速计算）"""
    return np.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2)


def find_nearest_hotels(origin_lat, origin_lng, target_lats, target_lngs, num_samples, origin_city=None, target_cities=None):
    """找到最近的酒店，优先选择城市名一致的数据"""
    # 快速过滤：经纬度差值超过阈值的直接排除
    lat_diff = np.abs(origin_lat - target_lats)
    lng_diff = np.abs(origin_lng - target_lngs)

    # 粗筛：只有当经纬度差值都小于阈值时才进行详细距离计算
    mask = (lat_diff < config['lat_lng_threshold']) & (
        lng_diff < config['lat_lng_threshold'])

    if not np.any(mask):
        # 如果没有通过粗筛的点，则只能使用所有点计算距离
        distances = calculate_distance(
            origin_lat, origin_lng, target_lats, target_lngs)
        city_match_mask = None
    else:
        # 对通过粗筛的点计算精确距离
        filtered_lats = target_lats[mask]
        filtered_lngs = target_lngs[mask]

        # 计算距离
        distances = np.full_like(target_lats, np.inf)
        filtered_distances = calculate_distance(
            origin_lat, origin_lng, filtered_lats, filtered_lngs)
        distances[mask] = filtered_distances

        # 如果提供了城市名信息，创建城市名匹配掩码
        if origin_city is not None and target_cities is not None:
            city_match_mask = np.full_like(target_lats, False, dtype=bool)
            filtered_cities = target_cities[mask]
            # 城市名匹配（忽略大小写和空格），处理NaN值
            city_match = np.array([
                (not pd.isna(origin_city) and not pd.isna(city) and
                 str(origin_city).lower().strip() == str(city).lower().strip())
                for city in filtered_cities
            ])
            city_match_mask[mask] = city_match
        else:
            city_match_mask = None

    # 获取距离最小的索引（不包括自身，自身距离为0）
    if city_match_mask is not None and np.any(city_match_mask):
        # 优先选择城市名一致的
        city_match_indices = np.where(city_match_mask)[0]
        city_match_distances = distances[city_match_indices]
        city_match_sorted = np.argsort(city_match_distances)
        city_match_nearest = city_match_indices[city_match_sorted][:num_samples]

        # 如果城市名一致的数量不够，再补充其他城市的
        if len(city_match_nearest) < num_samples:
            remaining_needed = num_samples - len(city_match_nearest)
            other_indices = np.where(~city_match_mask)[0]
            other_distances = distances[other_indices]
            other_sorted = np.argsort(other_distances)
            other_nearest = other_indices[other_sorted][:remaining_needed]
            nearest_indices = np.concatenate(
                [city_match_nearest, other_nearest])
        else:
            nearest_indices = city_match_nearest
    else:
        # 没有城市名信息或没有城市名匹配，按距离排序
        nearest_indices = np.argsort(distances)[:num_samples]

    return nearest_indices


def generate_negative_samples_batch(args):
    """批量生成负样本（基于地理位置）"""
    try:
        group_df, num_neg_samples = args

        # 转换为numpy数组提高性能
        supply_texts = group_df['SupplyText'].values
        target_texts = group_df['TargetText'].values
        country_codes = group_df['CountryCode'].values

        # 获取经纬度数据
        target_lats = group_df['TargetLat'].values
        target_lngs = group_df['TargetLng'].values

        # 获取城市名数据
        supply_cities = group_df['CityName'].values if 'CityName' in group_df.columns else None
        target_cities = group_df['MappingWorksCityName'].values if 'MappingWorksCityName' in group_df.columns else None

        if len(target_texts) <= 1:
            return None

        # 预分配数组
        total_samples = min(num_neg_samples, len(
            supply_texts) * (len(target_texts) - 1))
        supply_texts_arr = np.empty(total_samples, dtype=object)
        target_texts_arr = np.empty(total_samples, dtype=object)
        country_codes_arr = np.empty(total_samples, dtype=object)
        target_lats_arr = np.empty(total_samples, dtype=np.float64)
        target_lngs_arr = np.empty(total_samples, dtype=np.float64)

        idx = 0
        for supply_idx, (supply_text, original_target, country_code, original_target_lat, original_target_lng) in enumerate(
                zip(supply_texts, target_texts, country_codes, target_lats, target_lngs)):

            # 排除原始匹配的索引
            mask = target_texts != original_target
            if not np.any(mask):
                continue

            available_targets = target_texts[mask]
            available_lats = target_lats[mask]
            available_lngs = target_lngs[mask]

            # 获取当前供应商的城市名
            origin_city = supply_cities[supply_idx] if supply_cities is not None else None
            available_cities = target_cities[mask] if target_cities is not None else None

            # 计算当前供应商文本需要的负样本数
            current_samples = min(
                num_neg_samples // len(supply_texts) + (1 if supply_idx <
                                                        num_neg_samples % len(supply_texts) else 0),
                len(available_targets)
            )

            if current_samples > 0:
                # 使用原始目标方的经纬度找到最近的其他目标方作为负样本
                nearest_indices = find_nearest_hotels(
                    original_target_lat, original_target_lng,
                    available_lats, available_lngs,
                    current_samples,
                    origin_city, available_cities
                )

                selected_targets = available_targets[nearest_indices]
                selected_lats = available_lats[nearest_indices]
                selected_lngs = available_lngs[nearest_indices]

                end_idx = idx + current_samples
                supply_texts_arr[idx:end_idx] = supply_text
                target_texts_arr[idx:end_idx] = selected_targets
                country_codes_arr[idx:end_idx] = country_code
                target_lats_arr[idx:end_idx] = selected_lats
                target_lngs_arr[idx:end_idx] = selected_lngs
                idx = end_idx

        if idx > 0:
            # 创建DataFrame
            return pd.DataFrame({
                'SupplyText': supply_texts_arr[:idx],
                'TargetText': target_texts_arr[:idx],
                'CountryCode': country_codes_arr[:idx],
                'TargetLat': target_lats_arr[:idx],
                'TargetLng': target_lngs_arr[:idx],
                'label': np.zeros(idx, dtype=np.int8)
            })
        return None

    except Exception as e:
        logging.error(f'生成负样本时出错: {str(e)}')
        return None


def process_data_chunk(df_chunk):
    """处理数据块"""
    try:
        # 向量化创建文本格式
        df_chunk['SupplyText'] = create_text_format_vectorized(df_chunk, True)
        df_chunk['TargetText'] = create_text_format_vectorized(df_chunk, False)

        # 根据SupplyText和TargetText去重
        df_chunk = df_chunk.drop_duplicates(
            subset=['SupplyText'])

        df_chunk = df_chunk.drop_duplicates(
            subset=['TargetText'])

        # 添加标签列（正样本）
        df_chunk['label'] = 1

        # 保留需要的列，包括城市名字段
        columns_to_keep = ['SupplyText', 'TargetText',
                           'CountryCode', 'label', 'TargetLat', 'TargetLng']
        if 'CityName' in df_chunk.columns:
            columns_to_keep.append('CityName')
        if 'MappingWorksCityName' in df_chunk.columns:
            columns_to_keep.append('MappingWorksCityName')

        return df_chunk[columns_to_keep]
    except Exception as e:
        logging.error(f'处理数据块时出错: {str(e)}')
        return None


def process_data(df):
    """处理数据的主要逻辑"""
    try:
        # 分块处理数据
        chunks = [df[i:i + config['batch_size']]
                  for i in range(0, len(df), config['batch_size'])]
        processed_chunks = []

        logging.info(f'开始处理 {len(chunks)} 个数据块...')

        # 使用多进程处理数据块
        with mp.Pool(processes=config['n_processes'], initializer=init_worker) as pool:
            for processed_chunk in tqdm(
                pool.imap_unordered(process_data_chunk, chunks),
                total=len(chunks),
                desc="处理数据块"
            ):
                if processed_chunk is not None:
                    processed_chunks.append(processed_chunk)

        if not processed_chunks:
            logging.error('数据处理失败')
            return None

        # 合并处理后的数据块
        df_processed = pd.concat(processed_chunks, ignore_index=True)
        logging.info(f'处理后的数据大小: {len(df_processed)} 行')

        # 按国家分组并准备数据
        country_groups = []
        for _, group in df_processed.groupby('CountryCode'):
            if len(group) > 1:
                country_groups.append(
                    (group, len(group) * config['neg_pos_ratio']))

        logging.info(f'开始使用 {config["n_processes"]} 个进程生成负样本...')

        # 使用多进程处理负样本生成
        negative_samples = []
        with mp.Pool(processes=config['n_processes'], initializer=init_worker) as pool:
            for neg_sample in tqdm(
                pool.imap_unordered(
                    generate_negative_samples_batch, country_groups, chunksize=1),
                total=len(country_groups),
                desc="生成负样本"
            ):
                if neg_sample is not None:
                    negative_samples.append(neg_sample)

        if not negative_samples:
            logging.error('无法生成负样本')
            return None

        # 合并负样本
        negative_df = pd.concat(negative_samples, ignore_index=True)

        # 确保整体的正负样本比例
        total_neg_samples_needed = len(df_processed) * config['neg_pos_ratio']
        if len(negative_df) > total_neg_samples_needed:
            negative_df = negative_df.sample(n=int(total_neg_samples_needed))

        # 合并并打乱数据顺序
        final_df = pd.concat([df_processed, negative_df], ignore_index=True)
        final_df = final_df.sample(frac=1).reset_index(drop=True)

        # 保留所有需要的列，包括经纬度和城市名
        final_columns = ['SupplyText', 'TargetText',
                         'CountryCode', 'TargetLat', 'TargetLng', 'label']
        if 'CityName' in final_df.columns:
            final_columns.append('CityName')
        if 'MappingWorksCityName' in final_df.columns:
            final_columns.append('MappingWorksCityName')

        final_df = final_df[final_columns]

        return final_df

    except Exception as e:
        logging.error(f'处理数据时出错: {str(e)}')
        return None


def clean_float_column(series, column_name, remove_invalid=False):
    """清理float类型的列数据
    Args:
        series: 需要清理的数据列
        column_name: 列名（用于日志）
        remove_invalid: 是否移除无效值（True）或填充0（False）
    Returns:
        清理后的数据列和无效数据的索引（如果remove_invalid=True）
    """
    # 移除前后空格和不间断空格
    cleaned = series.str.strip() if isinstance(series, pd.Series) else series
    cleaned = pd.to_numeric(cleaned, errors='coerce')
    invalid_mask = cleaned.isna()
    invalid_count = invalid_mask.sum()

    if invalid_count > 0:
        if remove_invalid:
            logging.warning(f'{column_name} 列有 {invalid_count} 个无效值所在行将被移除')
            return cleaned, invalid_mask
        else:
            logging.warning(f'{column_name} 列有 {invalid_count} 个无效值被替换为0')
            return cleaned.fillna(0), None

    return cleaned, None


def main():
    if not Path(config['input_file']).exists():
        logging.error(f'输入文件不存在: {config["input_file"]}')
        return

    logging.info('开始处理数据...')

    # 读取CSV文件
    try:
        # 分块读取文件
        chunks = []
        total_rows = 0
        invalid_rows = 0

        for chunk in tqdm(
            pd.read_csv(
                config['input_file'],
                dtype={
                    'SourceKey': str,
                    'CountryCode': str,
                    'MappingWorksLat': str,  # 改为str先读入
                    'MappingWorksLng': str,  # 改为str先读入
                    'CityName': str,
                    'MappingWorksCityName': str,
                },
                chunksize=config['batch_size'],
                low_memory=False
            ),
            desc="读取数据"
        ):
            try:
                total_rows += len(chunk)

                # 处理经纬度数据，移除无效值所在的行
                lat_cleaned, lat_invalid = clean_float_column(
                    chunk['MappingWorksLat'], 'MappingWorksLat', remove_invalid=True)
                lng_cleaned, lng_invalid = clean_float_column(
                    chunk['MappingWorksLng'], 'MappingWorksLng', remove_invalid=True)

                # 合并经纬度无效的掩码
                invalid_mask = lat_invalid | lng_invalid if lat_invalid is not None and lng_invalid is not None else None

                if invalid_mask is not None:
                    # 记录被移除的行数
                    removed_rows = invalid_mask.sum()
                    invalid_rows += removed_rows

                    # 移除无效行
                    chunk = chunk[~invalid_mask].copy()
                    chunk['TargetLat'] = lat_cleaned[~invalid_mask]
                    chunk['TargetLng'] = lng_cleaned[~invalid_mask]
                else:
                    chunk['TargetLat'] = lat_cleaned
                    chunk['TargetLng'] = lng_cleaned

                # 移除所有列中的不间断空格
                for col in chunk.columns:
                    if chunk[col].dtype == 'object':
                        chunk[col] = chunk[col].str.replace(
                            '\xa0', ' ').str.strip()

                if len(chunk) > 0:  # 只添加还有数据的块
                    chunks.append(chunk)

            except Exception as e:
                invalid_rows += len(chunk)
                logging.warning(f'处理数据块时出现警告: {str(e)}')
                continue

        if not chunks:
            logging.error('没有有效数据被读取')
            return

        df = pd.concat(chunks, ignore_index=True)
        logging.info(
            f'读取原始数据完成，总行数: {total_rows}，有效行数: {len(df)}，无效行数: {invalid_rows}')

        # 分别处理有SourceKey和无SourceKey的数据
        df['has_source_key'] = df['SourceKey'].notna() & (df['SourceKey'] != '')
        df_with_key = df[df['has_source_key']].copy()
        df_without_key = df[~df['has_source_key']].copy()

        # 只对有SourceKey的数据进行去重
        df_with_key_dedup = df_with_key.drop_duplicates(subset=['SourceKey'])

        # 合并回去重后的数据
        df = pd.concat([df_with_key_dedup, df_without_key], ignore_index=True)
        df = df.drop('has_source_key', axis=1)  # 删除临时列

        logging.info(f'数据处理完成：')
        logging.info(
            f'  - 有SourceKey的数据：原始 {len(df_with_key)} 行，去重后 {len(df_with_key_dedup)} 行')
        logging.info(f'  - 无SourceKey的数据：{len(df_without_key)} 行')
        logging.info(f'  - 最终数据总量：{len(df)} 行')

    except Exception as e:
        logging.error(f'读取文件时出错: {str(e)}')
        return

    # 设置随机种子
    np.random.seed(config['random_seed'])

    # 处理数据
    final_df = process_data(df)

    if final_df is not None:
        # 保存结果
        try:
            # 分块保存结果
            chunk_size = config['batch_size']
            num_chunks = (len(final_df) + chunk_size - 1) // chunk_size

            with open(config['output_file'], 'w', encoding='utf-8') as f:
                # 写入表头（包含经纬度和城市名字段）
                header_columns = ['SupplyText', 'TargetText',
                                  'CountryCode', 'TargetLat', 'TargetLng', 'label']
                if 'CityName' in final_df.columns:
                    header_columns.append('CityName')
                if 'MappingWorksCityName' in final_df.columns:
                    header_columns.append('MappingWorksCityName')
                f.write(','.join(header_columns) + '\n')

            for i in tqdm(range(num_chunks), desc="保存数据"):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(final_df))
                chunk = final_df.iloc[start_idx:end_idx]

                # 追加模式写入数据
                chunk.to_csv(config['output_file'], mode='a',
                             header=False, index=False, encoding='utf-8')

            logging.info(f'处理完成！共生成 {len(final_df)} 条数据')
            logging.info(
                f'其中正样本: {len(final_df[final_df["label"]==1])} 条，负样本: {len(final_df[final_df["label"]==0])} 条')
            logging.info(
                f'正负样本比例: 1:{len(final_df[final_df["label"]==0])/len(final_df[final_df["label"]==1]):.2f}')
            logging.info(f'数据已保存到: {config["output_file"]}')
        except Exception as e:
            logging.error(f'保存文件时出错: {str(e)}')
    else:
        logging.error('数据处理失败')


if __name__ == '__main__':
    main()
