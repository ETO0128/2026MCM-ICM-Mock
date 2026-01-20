import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

# =================配置区域=================
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 常量配置
AVG_PERSON_WEIGHT = 70  # 假设平均每人70kg
TIME_SLOT_MINUTES = 5  # 时间分析粒度（分钟）


# =========================================

def get_data_path():
    """智能定位data文件夹"""
    current_dir = Path(__file__).parent.absolute()

    # 尝试多种可能的路径
    possible_paths = [
        current_dir / 'data',
        current_dir,
        current_dir.parent / 'data',
        Path.cwd() / 'data'
    ]

    for path in possible_paths:
        if path.exists() and any(path.glob('*.csv')):
            return path

    raise FileNotFoundError(f"❌ 找不到数据文件夹")


def load_and_clean(file_path, cols=None, parse_dates=['Time']):
    """
    读取CSV文件，自动处理中文编码
    """
    print(f"正在读取: {file_path.name}")

    encodings = ['gb18030', 'gbk', 'utf-8-sig', 'ansi', 'utf-8']

    for enc in encodings:
        try:
            if cols:
                df = pd.read_csv(file_path, usecols=cols, encoding=enc)
            else:
                df = pd.read_csv(file_path, encoding=enc)
            print(f"  -> 成功使用 [{enc}] 编码")

            # 清理列名
            df.columns = df.columns.str.strip()

            # 如果有Time列，转换为datetime类型
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
                # 删除时间解析失败的行
                df = df.dropna(subset=['Time'])
                df = df.sort_values('Time').reset_index(drop=True)

            # 标准化数据类型
            if 'Floor' in df.columns:
                # 尝试将Floor转换为数值类型
                df['Floor'] = pd.to_numeric(df['Floor'], errors='coerce')
                df = df.dropna(subset=['Floor'])
                df['Floor'] = df['Floor'].astype(int)

            if 'Elevator ID' in df.columns:
                df['Elevator ID'] = df['Elevator ID'].astype(str).str.strip()

            return df

        except Exception as e:
            continue

    print(f"❌ 无法读取 {file_path.name}")
    return None


def estimate_passenger_count(load_changes_df):
    """
    根据重量变化估算乘客数量
    """
    if load_changes_df is None or load_changes_df.empty:
        return pd.DataFrame()

    df = load_changes_df.copy()

    # 确保数值列是数值类型
    for col in ['Load In (kg)', 'Load Out (kg)']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 计算进出乘客数
    df['Passengers_In'] = df['Load In (kg)'] / AVG_PERSON_WEIGHT
    df['Passengers_Out'] = df['Load Out (kg)'] / AVG_PERSON_WEIGHT

    # 四舍五入到最近的整数
    df['Passengers_In'] = df['Passengers_In'].round().astype(int)
    df['Passengers_Out'] = df['Passengers_Out'].round().astype(int)
    df['Net_Passengers'] = df['Passengers_In'] - df['Passengers_Out']

    # 添加时间特征
    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute
    df['Time_Slot'] = df['Time'].dt.floor(f'{TIME_SLOT_MINUTES}min')

    return df


def calculate_wait_times_simple(hall_calls, car_stops):
    """
    简化的等待时间计算方法
    """
    print("\n[计算等待时间 - 简化方法]")

    if hall_calls.empty or car_stops.empty:
        print("❌ 数据为空，无法计算等待时间")
        return pd.DataFrame()

    # 确保数据类型一致
    hall_calls = hall_calls.copy()
    car_stops = car_stops.copy()

    # 标准化数据类型
    hall_calls['Floor'] = hall_calls['Floor'].astype(int)
    car_stops['Floor'] = car_stops['Floor'].astype(int)

    # 确保数据已排序
    hall_calls = hall_calls.sort_values('Time')
    car_stops = car_stops.sort_values('Time')

    # 为每个呼叫寻找匹配的停靠
    wait_times = []

    # 按电梯分组处理
    for elevator in hall_calls['Elevator ID'].unique():
        hall_elev = hall_calls[hall_calls['Elevator ID'] == elevator]
        stop_elev = car_stops[car_stops['Elevator ID'] == elevator]

        if hall_elev.empty or stop_elev.empty:
            continue

        # 对每个呼叫，找到同楼层、同方向、时间最近的停靠
        for _, hall_row in hall_elev.iterrows():
            call_time = hall_row['Time']
            floor = hall_row['Floor']
            direction = hall_row['Direction']

            # 找到符合条件的停靠
            matching = stop_elev[
                (stop_elev['Floor'] == floor) &
                (stop_elev['Direction'] == direction) &
                (stop_elev['Time'] >= call_time)
                ]

            if not matching.empty:
                stop_time = matching.iloc[0]['Time']
                wait_seconds = (stop_time - call_time).total_seconds()

                # 过滤合理范围
                if 1 <= wait_seconds <= 900:
                    wait_times.append({
                        'Time_call': call_time,
                        'Floor': floor,
                        'Direction': direction,
                        'Elevator_ID': elevator,
                        'Time_stop': stop_time,
                        'Wait_Time': wait_seconds
                    })

    if wait_times:
        result = pd.DataFrame(wait_times)
        avg_wait = result['Wait_Time'].mean()
        print(f"✅ 平均等待时间: {avg_wait:.2f}秒")
        print(f"✅ 有效等待记录: {len(result)}条")
        return result
    else:
        print("❌ 未能计算等待时间")
        return pd.DataFrame()


def analyze_traffic_patterns(hall_calls, time_slot_minutes=5):
    """
    按时间槽分析流量模式
    """
    print(f"\n[流量模式分析] 时间粒度: {time_slot_minutes}分钟")

    if hall_calls.empty:
        print("❌ 无大厅呼叫数据")
        return pd.DataFrame()

    df = hall_calls.copy()

    # 创建时间槽
    df['Time_Slot'] = df['Time'].dt.floor(f'{time_slot_minutes}min')

    # 按时间槽统计
    time_slot_stats = df.groupby('Time_Slot').agg({
        'Floor': 'count',  # 呼叫次数
    }).rename(columns={'Floor': 'Call_Count'})

    # 统计上行下行比例
    up_counts = df[df['Direction'] == 'Up'].groupby('Time_Slot').size()
    down_counts = df[df['Direction'] == 'Down'].groupby('Time_Slot').size()

    time_slot_stats['Up_Count'] = time_slot_stats.index.map(lambda x: up_counts.get(x, 0))
    time_slot_stats['Down_Count'] = time_slot_stats.index.map(lambda x: down_counts.get(x, 0))

    # 计算上行比例
    time_slot_stats['Up_Ratio'] = time_slot_stats.apply(
        lambda row: row['Up_Count'] / row['Call_Count'] if row['Call_Count'] > 0 else 0,
        axis=1
    )

    # 添加小时和分钟信息
    time_slot_stats['Hour'] = time_slot_stats.index.hour
    time_slot_stats['Minute'] = time_slot_stats.index.minute

    print(f"✅ 流量模式分析完成: {len(time_slot_stats)}个时间槽")
    return time_slot_stats


def analyze_floor_demand(hall_calls, car_calls):
    """
    分析楼层需求（作为起点和终点）
    """
    print("\n[楼层需求分析]")

    # 作为起点的楼层（大厅呼叫）
    if hall_calls.empty:
        start_floors = pd.Series(dtype=int)
    else:
        start_floors = hall_calls['Floor'].value_counts().sort_index()

    # 作为终点的楼层（轿厢呼叫）
    if car_calls is None or car_calls.empty:
        end_floors = pd.Series(dtype=int)
    else:
        # 只考虑注册的呼叫
        if 'Action' in car_calls.columns:
            registered_calls = car_calls[car_calls['Action'] == 'Register']
            end_floors = registered_calls['Floor'].value_counts().sort_index()
        else:
            end_floors = car_calls['Floor'].value_counts().sort_index()

    print(f"✅ 起点楼层分析: {len(start_floors)}个楼层")
    if not end_floors.empty:
        print(f"✅ 终点楼层分析: {len(end_floors)}个楼层")

    return start_floors, end_floors


def classify_traffic_mode(time_slot_stats):
    """
    根据流量特征分类交通模式
    """
    if time_slot_stats.empty:
        return time_slot_stats

    print("\n[交通模式分类]")

    modes = []

    for idx, row in time_slot_stats.iterrows():
        hour = row['Hour']
        up_ratio = row['Up_Ratio']
        call_count = row['Call_Count']

        # 根据规则分类
        if call_count == 0:
            mode = '无流量'
        elif call_count <= 1:
            mode = '极低流量'
        elif 7 <= hour < 9 and up_ratio > 0.7:
            mode = '早晨上行高峰'
        elif 17 <= hour < 19 and up_ratio < 0.3:
            mode = '晚间下行高峰'
        elif 11 <= hour < 13 and 0.4 <= up_ratio <= 0.6:
            mode = '午餐时段'
        elif call_count >= 5:
            mode = '高流量'
        else:
            mode = '正常流量'

        modes.append(mode)

    time_slot_stats['Traffic_Mode'] = modes

    # 统计各模式占比
    mode_counts = time_slot_stats['Traffic_Mode'].value_counts()
    for mode, count in mode_counts.items():
        percentage = count / len(time_slot_stats) * 100
        print(f"  {mode}: {count}个时间槽 ({percentage:.1f}%)")

    return time_slot_stats


def generate_statistics_report(data_frames, output_path):
    """
    生成详细的统计报告 - 修复版，处理Series对象
    """
    print(f"\n[生成统计报告] {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("电梯系统运行统计分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        # 1. 数据集概览
        f.write("1. 数据集概览\n")
        f.write("-" * 40 + "\n")

        # 只处理DataFrame对象
        for name, data in data_frames.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                f.write(f"{name}:\n")
                f.write(f"  记录数: {len(data):,}\n")

                if 'Time' in data.columns:
                    f.write(f"  时间范围: {data['Time'].min()} 到 {data['Time'].max()}\n")
                    # 计算天数
                    time_range = data['Time'].max() - data['Time'].min()
                    f.write(f"  天数: {time_range.days + 1}天\n")

                if 'Elevator ID' in data.columns:
                    elevators = data['Elevator ID'].unique()
                    f.write(f"  电梯数量: {len(elevators)}\n")

                if 'Floor' in data.columns:
                    floors = data['Floor'].unique()
                    if len(floors) > 0:
                        f.write(f"  涉及楼层: {len(floors)}层 (最低{min(floors)}, 最高{max(floors)})\n")

                f.write("\n")
            elif isinstance(data, pd.Series) and not data.empty:
                f.write(f"{name} (统计序列):\n")
                f.write(f"  条目数: {len(data):,}\n")
                f.write(f"  总计: {data.sum():,}\n")
                f.write("\n")

        # 2. 等待时间分析
        if 'wait_times' in data_frames and isinstance(data_frames['wait_times'], pd.DataFrame) and not data_frames[
            'wait_times'].empty:
            wait_df = data_frames['wait_times']
            f.write("2. 等待时间分析\n")
            f.write("-" * 40 + "\n")
            f.write(f"总记录数: {len(wait_df):,}\n")
            f.write(f"平均等待时间: {wait_df['Wait_Time'].mean():.2f}秒\n")
            f.write(f"中位数等待时间: {wait_df['Wait_Time'].median():.2f}秒\n")
            f.write(f"标准差: {wait_df['Wait_Time'].std():.2f}秒\n")

            # 百分位数
            percentiles = [25, 50, 75, 90, 95]
            for p in percentiles:
                value = wait_df['Wait_Time'].quantile(p / 100)
                f.write(f"{p}百分位数: {value:.1f}秒\n")

            # 长等待统计（超过60秒）
            long_waits = wait_df[wait_df['Wait_Time'] > 60]
            if len(long_waits) > 0:
                percentage = len(long_waits) / len(wait_df) * 100
                f.write(f"长等待(>60秒)比例: {percentage:.1f}% ({len(long_waits)}次)\n")
                f.write(f"最长等待: {wait_df['Wait_Time'].max():.1f}秒\n")

            # 按电梯统计
            if 'Elevator_ID' in wait_df.columns:
                f.write("\n按电梯统计:\n")
                for elevator in sorted(wait_df['Elevator_ID'].unique()):
                    elev_waits = wait_df[wait_df['Elevator_ID'] == elevator]['Wait_Time']
                    f.write(f"  电梯{elevator}: {elev_waits.mean():.1f}秒 (N={len(elev_waits)})\n")

            f.write("\n")

        # 3. 流量模式分析
        if 'time_slot_stats' in data_frames and isinstance(data_frames['time_slot_stats'], pd.DataFrame) and not \
        data_frames['time_slot_stats'].empty:
            ts_stats = data_frames['time_slot_stats']
            f.write("3. 流量模式分析\n")
            f.write("-" * 40 + "\n")
            f.write(f"时间分析粒度: {TIME_SLOT_MINUTES}分钟\n")
            f.write(f"总时间槽数: {len(ts_stats)}\n")
            f.write(f"总呼叫次数: {ts_stats['Call_Count'].sum():,}\n")
            f.write(f"平均每槽呼叫数: {ts_stats['Call_Count'].mean():.2f}\n")

            # 高峰时段识别
            if not ts_stats.empty:
                top_slots = ts_stats.nlargest(5, 'Call_Count')
                f.write("\n高峰时段(前5):\n")
                for idx, row in top_slots.iterrows():
                    f.write(f"  {idx.strftime('%H:%M')}: {row['Call_Count']}次呼叫 "
                            f"(上行{row['Up_Count']}/下行{row['Down_Count']})\n")

            f.write("\n")

        # 4. 楼层需求分析
        if 'start_floors' in data_frames:
            f.write("4. 楼层需求分析\n")
            f.write("-" * 40 + "\n")

            start_floors = data_frames['start_floors']

            if isinstance(start_floors, pd.Series) and not start_floors.empty:
                f.write("作为起点的热门楼层(前10):\n")
                total_calls = start_floors.sum()
                for floor, count in start_floors.head(10).items():
                    percentage = count / total_calls * 100
                    f.write(f"  楼层{floor}: {count}次 ({percentage:.1f}%)\n")

            if 'end_floors' in data_frames and isinstance(data_frames['end_floors'], pd.Series) and not data_frames[
                'end_floors'].empty:
                end_floors = data_frames['end_floors']
                f.write("\n作为终点的热门楼层(前10):\n")
                total_calls = end_floors.sum()
                for floor, count in end_floors.head(10).items():
                    percentage = count / total_calls * 100
                    f.write(f"  楼层{floor}: {count}次 ({percentage:.1f}%)\n")

            f.write("\n")

        # 5. 乘客流量估算
        if 'passenger_flow' in data_frames and isinstance(data_frames['passenger_flow'], pd.DataFrame) and not \
        data_frames['passenger_flow'].empty:
            pass_df = data_frames['passenger_flow']
            f.write("5. 乘客流量估算\n")
            f.write("-" * 40 + "\n")
            f.write(f"总记录数: {len(pass_df):,}\n")
            f.write(f"估算总进客数: {pass_df['Passengers_In'].sum():.0f}人\n")
            f.write(f"估算总出客数: {pass_df['Passengers_Out'].sum():.0f}人\n")
            f.write(f"净变化: {pass_df['Net_Passengers'].sum():.0f}人\n")

            # 按小时统计
            if 'Hour' in pass_df.columns:
                hourly_passengers = pass_df.groupby('Hour').agg({
                    'Passengers_In': 'sum',
                    'Passengers_Out': 'sum'
                })

                if not hourly_passengers.empty:
                    f.write("\n每小时乘客流量:\n")
                    for hour in range(24):
                        if hour in hourly_passengers.index:
                            in_count = hourly_passengers.loc[hour, 'Passengers_In']
                            out_count = hourly_passengers.loc[hour, 'Passengers_Out']
                            f.write(f"  {hour:02d}:00 - {in_count:.0f}人进 / {out_count:.0f}人出\n")

            f.write("\n")

        # 6. 交通模式分布
        if 'time_slot_stats' in data_frames and isinstance(data_frames['time_slot_stats'],
                                                           pd.DataFrame) and 'Traffic_Mode' in data_frames[
            'time_slot_stats'].columns:
            ts_stats = data_frames['time_slot_stats']
            f.write("6. 交通模式分布\n")
            f.write("-" * 40 + "\n")

            mode_dist = ts_stats['Traffic_Mode'].value_counts()
            for mode, count in mode_dist.items():
                percentage = count / len(ts_stats) * 100
                avg_calls = ts_stats[ts_stats['Traffic_Mode'] == mode]['Call_Count'].mean()
                f.write(f"{mode}: {count}槽 ({percentage:.1f}%), 平均呼叫数: {avg_calls:.2f}\n")

        # 7. 总结与建议
        f.write("\n7. 总结与建议\n")
        f.write("-" * 40 + "\n")

        # 基于分析结果提供建议
        if 'wait_times' in data_frames and isinstance(data_frames['wait_times'], pd.DataFrame) and not data_frames[
            'wait_times'].empty:
            avg_wait = data_frames['wait_times']['Wait_Time'].mean()
            if avg_wait > 60:
                f.write(f"⚠️  平均等待时间({avg_wait:.1f}秒)偏高，建议优化调度策略\n")
            elif avg_wait > 40:
                f.write(f"📊 平均等待时间({avg_wait:.1f}秒)可接受，但仍有优化空间\n")
            else:
                f.write(f"✅ 平均等待时间({avg_wait:.1f}秒)表现良好\n")

        if 'time_slot_stats' in data_frames and isinstance(data_frames['time_slot_stats'], pd.DataFrame) and not \
        data_frames['time_slot_stats'].empty:
            # 识别高峰时段
            peak_hours = []
            ts_stats = data_frames['time_slot_stats']
            for hour in range(6, 22):  # 6点到22点
                hour_calls = ts_stats[ts_stats['Hour'] == hour]['Call_Count'].sum()
                if hour_calls > ts_stats['Call_Count'].mean() * 2:  # 超过平均2倍
                    peak_hours.append(hour)

            if peak_hours:
                f.write(f"🚀 识别到高峰时段: {', '.join([f'{h}:00' for h in peak_hours])}\n")
                f.write("   建议在高峰时段增加电梯调度频率或预置电梯\n")

        if 'start_floors' in data_frames and isinstance(data_frames['start_floors'], pd.Series) and not data_frames[
            'start_floors'].empty:
            top_floor = data_frames['start_floors'].idxmax()
            top_count = data_frames['start_floors'].max()
            f.write(f"📍 最热门的起点楼层: {top_floor}层 ({top_count}次呼叫)\n")
            f.write(f"   建议将空闲电梯预置在该楼层附近\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("报告结束\n")

    print(f"✅ 统计报告已保存到: {output_path}")


def create_visualizations(data_frames, results_dir):
    """创建可视化图表"""
    print("\n[生成可视化图表]")

    try:
        # 创建第一个图表：关键指标
        plt.figure(figsize=(15, 10))

        # 子图1: 全天流量曲线
        plt.subplot(2, 3, 1)
        if 'time_slot_stats' in data_frames and isinstance(data_frames['time_slot_stats'], pd.DataFrame) and not \
        data_frames['time_slot_stats'].empty:
            ts_stats = data_frames['time_slot_stats']
            # 按小时聚合
            hourly_stats = ts_stats.groupby('Hour')['Call_Count'].sum()
            plt.plot(hourly_stats.index, hourly_stats.values, marker='o', linewidth=2, color='steelblue')
            plt.title(f'每小时呼叫量 ({TIME_SLOT_MINUTES}分钟粒度)', fontsize=12)
            plt.xlabel('小时')
            plt.ylabel('呼叫次数')
            plt.xticks(range(0, 24, 2))
            plt.grid(True, alpha=0.3)
            plt.fill_between(hourly_stats.index, 0, hourly_stats.values, alpha=0.3, color='steelblue')

        # 子图2: 等待时间分布
        plt.subplot(2, 3, 2)
        if 'wait_times' in data_frames and isinstance(data_frames['wait_times'], pd.DataFrame) and not data_frames[
            'wait_times'].empty:
            wait_df = data_frames['wait_times']
            if not wait_df.empty:
                plt.hist(wait_df['Wait_Time'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                plt.title('等待时间分布', fontsize=12)
                plt.xlabel('等待时间 (秒)')
                plt.ylabel('频次')
                max_wait = min(300, wait_df['Wait_Time'].max() * 1.1)
                plt.xlim(0, max_wait)
                mean_wait = wait_df['Wait_Time'].mean()
                median_wait = wait_df['Wait_Time'].median()
                plt.axvline(mean_wait, color='red', linestyle='--', label=f'平均: {mean_wait:.1f}s')
                plt.axvline(median_wait, color='green', linestyle='--', label=f'中位数: {median_wait:.1f}s')
                plt.legend(fontsize=9)

        # 子图3: 起点楼层热度
        plt.subplot(2, 3, 3)
        if 'start_floors' in data_frames and isinstance(data_frames['start_floors'], pd.Series) and not data_frames[
            'start_floors'].empty:
            start_floors = data_frames['start_floors']
            top_10 = start_floors.head(10)
            if len(top_10) > 0:
                colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_10)))
                plt.bar(range(len(top_10)), top_10.values, color=colors, alpha=0.7)
                plt.title('起点楼层热度 (Top 10)', fontsize=12)
                plt.xlabel('楼层')
                plt.ylabel('呼叫次数')
                plt.xticks(range(len(top_10)), top_10.index, rotation=45)

        # 子图4: 各时段上行比例
        plt.subplot(2, 3, 4)
        if 'time_slot_stats' in data_frames and isinstance(data_frames['time_slot_stats'], pd.DataFrame) and not \
        data_frames['time_slot_stats'].empty:
            ts_stats = data_frames['time_slot_stats']
            # 按小时计算平均上行比例
            hourly_up_ratio = ts_stats.groupby('Hour')['Up_Ratio'].mean()
            plt.bar(hourly_up_ratio.index, hourly_up_ratio.values,
                    color='orange', alpha=0.7, width=0.8)
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            plt.title('各小时上行呼叫比例', fontsize=12)
            plt.xlabel('小时')
            plt.ylabel('上行比例')
            plt.xticks(range(0, 24, 2))
            plt.ylim(0, 1)

        # 子图5: 交通模式分布
        plt.subplot(2, 3, 5)
        if 'time_slot_stats' in data_frames and isinstance(data_frames['time_slot_stats'],
                                                           pd.DataFrame) and 'Traffic_Mode' in data_frames[
            'time_slot_stats'].columns:
            mode_dist = data_frames['time_slot_stats']['Traffic_Mode'].value_counts()
            if len(mode_dist) > 0:
                colors = plt.cm.Set3(np.linspace(0, 1, len(mode_dist)))
                plt.pie(mode_dist.values, labels=mode_dist.index, autopct='%1.1f%%',
                        colors=colors, startangle=90, textprops={'fontsize': 9})
                plt.title('交通模式分布', fontsize=12)

        # 子图6: 乘客流量估算
        plt.subplot(2, 3, 6)
        if 'passenger_flow' in data_frames and isinstance(data_frames['passenger_flow'], pd.DataFrame) and not \
        data_frames['passenger_flow'].empty:
            pass_df = data_frames['passenger_flow']
            if 'Hour' in pass_df.columns:
                hourly_pass = pass_df.groupby('Hour').agg({
                    'Passengers_In': 'sum',
                    'Passengers_Out': 'sum'
                })
                if not hourly_pass.empty:
                    width = 0.35
                    x = np.arange(len(hourly_pass))
                    plt.bar(x - width / 2, hourly_pass['Passengers_In'], width,
                            label='进入', color='lightblue', alpha=0.7)
                    plt.bar(x + width / 2, hourly_pass['Passengers_Out'], width,
                            label='离开', color='lightcoral', alpha=0.7)
                    plt.title('每小时乘客进出估算', fontsize=12)
                    plt.xlabel('小时')
                    plt.ylabel('乘客数')
                    plt.xticks(x, hourly_pass.index)
                    plt.legend(fontsize=9)

        plt.tight_layout()
        save_path = results_dir / 'elevator_analysis_1.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 分析图表1已保存: {save_path}")
        plt.show()

    except Exception as e:
        print(f"❌ 创建图表1时出错: {e}")
        import traceback
        traceback.print_exc()

    try:
        # 创建第二个图表：详细分析
        plt.figure(figsize=(15, 8))

        # 子图1: 各电梯等待时间对比
        plt.subplot(2, 3, 1)
        if 'wait_times' in data_frames and isinstance(data_frames['wait_times'], pd.DataFrame) and not data_frames[
            'wait_times'].empty:
            wait_df = data_frames['wait_times']
            if 'Elevator_ID' in wait_df.columns:
                elev_means = wait_df.groupby('Elevator_ID')['Wait_Time'].mean().sort_values()
                if not elev_means.empty:
                    colors = plt.cm.coolwarm(np.linspace(0, 1, len(elev_means)))
                    bars = plt.bar(range(len(elev_means)), elev_means.values, color=colors, alpha=0.7)
                    plt.title('各电梯平均等待时间', fontsize=12)
                    plt.xlabel('电梯ID')
                    plt.ylabel('平均等待时间 (秒)')
                    plt.xticks(range(len(elev_means)), elev_means.index)
                    # 在柱状图上添加数值
                    for i, v in enumerate(elev_means.values):
                        plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

        # 子图2: 时间槽呼叫量热力图
        plt.subplot(2, 3, 2)
        if 'time_slot_stats' in data_frames and isinstance(data_frames['time_slot_stats'], pd.DataFrame) and not \
        data_frames['time_slot_stats'].empty:
            ts_stats = data_frames['time_slot_stats'].copy()
            # 限制显示的小时范围
            ts_stats = ts_stats[ts_stats['Hour'].between(6, 22)]  # 只显示6点到22点
            if not ts_stats.empty:
                # 创建小时-分钟的热力图数据
                heatmap_data = pd.pivot_table(
                    ts_stats.reset_index(),
                    values='Call_Count',
                    index='Hour',
                    columns='Minute',
                    aggfunc='mean',
                    fill_value=0
                )
                # 确保分钟列完整
                all_minutes = list(range(0, 60, TIME_SLOT_MINUTES))
                for minute in all_minutes:
                    if minute not in heatmap_data.columns:
                        heatmap_data[minute] = 0
                heatmap_data = heatmap_data[all_minutes]

                sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': '平均呼叫次数'})
                plt.title(f'{TIME_SLOT_MINUTES}分钟槽呼叫量热力图 (6:00-22:00)', fontsize=12)
                plt.xlabel('分钟')
                plt.ylabel('小时')

        # 子图3: 等待时间箱线图（按小时）
        plt.subplot(2, 3, 3)
        if 'wait_times' in data_frames and isinstance(data_frames['wait_times'], pd.DataFrame) and not data_frames[
            'wait_times'].empty:
            wait_df = data_frames['wait_times']
            wait_df['Hour'] = wait_df['Time_call'].dt.hour
            # 过滤异常值
            filtered = wait_df[(wait_df['Wait_Time'] >= 0) & (wait_df['Wait_Time'] <= 300)]
            if not filtered.empty:
                box_data = [filtered[filtered['Hour'] == h]['Wait_Time'].values for h in range(24)]
                positions = range(24)
                box = plt.boxplot(box_data, positions=positions, widths=0.6,
                                  patch_artist=True, showfliers=False)
                # 设置箱体颜色
                for patch in box['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                plt.title('各小时等待时间分布', fontsize=12)
                plt.xlabel('小时')
                plt.ylabel('等待时间 (秒)')
                plt.xticks(range(0, 24, 2))
                plt.ylim(0, min(300, filtered['Wait_Time'].max() * 1.1))

        # 子图4: 累计呼叫量
        plt.subplot(2, 3, 4)
        if 'hall_calls' in data_frames and isinstance(data_frames['hall_calls'], pd.DataFrame) and not data_frames[
            'hall_calls'].empty:
            hall_df = data_frames['hall_calls'].copy()
            hall_df = hall_df.sort_values('Time')
            hall_df['Cumulative_Calls'] = range(1, len(hall_df) + 1)
            plt.plot(hall_df['Time'], hall_df['Cumulative_Calls'], linewidth=2, color='darkgreen')
            plt.title('累计呼叫量随时间变化', fontsize=12)
            plt.xlabel('时间')
            plt.ylabel('累计呼叫次数')
            plt.grid(True, alpha=0.3)

        # 子图5: 各模式呼叫强度
        plt.subplot(2, 3, 5)
        if 'time_slot_stats' in data_frames and isinstance(data_frames['time_slot_stats'],
                                                           pd.DataFrame) and 'Traffic_Mode' in data_frames[
            'time_slot_stats'].columns:
            ts_stats = data_frames['time_slot_stats']
            mode_avg = ts_stats.groupby('Traffic_Mode')['Call_Count'].mean().sort_values(ascending=False)
            if not mode_avg.empty:
                colors = plt.cm.Paired(np.linspace(0, 1, len(mode_avg)))
                bars = plt.bar(range(len(mode_avg)), mode_avg.values, color=colors, alpha=0.7)
                plt.title('各交通模式平均呼叫强度', fontsize=12)
                plt.xlabel('交通模式')
                plt.ylabel('平均呼叫次数/槽')
                plt.xticks(range(len(mode_avg)), mode_avg.index, rotation=45, ha='right')
                # 添加数值标签
                for i, v in enumerate(mode_avg.values):
                    plt.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

        # 子图6: 等待时间与呼叫量关系
        plt.subplot(2, 3, 6)
        if 'wait_times' in data_frames and isinstance(data_frames['wait_times'], pd.DataFrame) and not data_frames[
            'wait_times'].empty and \
                'time_slot_stats' in data_frames and isinstance(data_frames['time_slot_stats'], pd.DataFrame) and not \
        data_frames['time_slot_stats'].empty:
            wait_df = data_frames['wait_times']
            ts_stats = data_frames['time_slot_stats']

            # 按时间槽对齐数据
            wait_df['Time_Slot'] = wait_df['Time_call'].dt.floor(f'{TIME_SLOT_MINUTES}min')
            wait_by_slot = wait_df.groupby('Time_Slot')['Wait_Time'].mean()
            calls_by_slot = ts_stats['Call_Count']

            # 找到共同的时间槽
            common_slots = wait_by_slot.index.intersection(calls_by_slot.index)
            if len(common_slots) > 0:
                wait_values = wait_by_slot.loc[common_slots].values
                call_values = calls_by_slot.loc[common_slots].values

                plt.scatter(call_values, wait_values, alpha=0.6, color='purple', s=30)

                # 添加趋势线
                if len(common_slots) > 1:
                    z = np.polyfit(call_values, wait_values, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(call_values), max(call_values), 100)
                    plt.plot(x_range, p(x_range), 'r--', alpha=0.8,
                             label=f'趋势线: y={z[0]:.2f}x+{z[1]:.2f}')
                    plt.legend(fontsize=9)

                plt.title('等待时间与呼叫量关系', fontsize=12)
                plt.xlabel('时间槽呼叫次数')
                plt.ylabel('平均等待时间 (秒)')
                plt.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = results_dir / 'elevator_analysis_2.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 分析图表2已保存: {save_path}")
        plt.show()

    except Exception as e:
        print(f"❌ 创建图表2时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("=" * 60)
    print("电梯数据分析系统")
    print("=" * 60)

    # 1. 获取数据路径
    try:
        data_dir = get_data_path()
        print(f"📂 数据目录: {data_dir}")
    except Exception as e:
        print(e)
        return

    # 2. 读取所有数据文件
    print("\n" + "=" * 60)
    print("加载数据文件")
    print("=" * 60)

    # 定义要加载的文件
    files_to_load = [
        ('hall_calls', 'hall_calls.csv', ['Time', 'Elevator ID', 'Direction', 'Floor']),
        ('car_calls', 'car_calls.csv', ['Time', 'Elevator ID', 'Floor', 'Action']),
        ('car_stops', 'car_stops.csv', ['Time', 'Elevator ID', 'Floor', 'Direction']),
        ('load_changes', 'load_changes.csv', ['Time', 'Elevator ID', 'Floor', 'Load In (kg)', 'Load Out (kg)']),
        ('car_departures', 'car_departures.csv', ['Time', 'Elevator ID', 'Floor']),
        ('maintenance_mode', 'maintenance_mode.csv', ['Time', 'Elevator ID', 'Action'])
    ]

    data_frames = {}
    for name, file_name, cols in files_to_load:
        file_path = data_dir / file_name
        if file_path.exists():
            df = load_and_clean(file_path, cols=cols)
            if df is not None:
                data_frames[name] = df
                print(f"✅ {name}: {len(df)} 条记录")
            else:
                print(f"❌ {name}: 读取失败")
                data_frames[name] = pd.DataFrame()
        else:
            print(f"⚠️  {name}: 文件不存在")
            data_frames[name] = pd.DataFrame()

    # 3. 数据分析和处理
    print("\n" + "=" * 60)
    print("数据分析处理")
    print("=" * 60)

    # 3.1 估算乘客流量
    if 'load_changes' in data_frames:
        passenger_flow = estimate_passenger_count(data_frames['load_changes'])
        if not passenger_flow.empty:
            print(f"✅ 乘客流量估算: {len(passenger_flow)} 条记录")
            data_frames['passenger_flow'] = passenger_flow
        else:
            print("⚠️  无法估算乘客流量")

    # 3.2 计算等待时间（使用简化方法）
    if 'hall_calls' in data_frames and 'car_stops' in data_frames:
        wait_times = calculate_wait_times_simple(data_frames['hall_calls'], data_frames['car_stops'])
        if not wait_times.empty:
            data_frames['wait_times'] = wait_times
        else:
            print("⚠️  无法计算等待时间")

    # 3.3 分析流量模式
    if 'hall_calls' in data_frames:
        time_slot_stats = analyze_traffic_patterns(data_frames['hall_calls'], TIME_SLOT_MINUTES)
        if not time_slot_stats.empty:
            data_frames['time_slot_stats'] = time_slot_stats

            # 3.4 分类交通模式
            time_slot_stats = classify_traffic_mode(time_slot_stats)
        else:
            print("⚠️  无法分析流量模式")

    # 3.5 分析楼层需求
    start_floors, end_floors = analyze_floor_demand(
        data_frames.get('hall_calls', pd.DataFrame()),
        data_frames.get('car_calls', pd.DataFrame())
    )
    data_frames['start_floors'] = start_floors
    data_frames['end_floors'] = end_floors

    # 4. 生成统计报告
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    report_path = results_dir / 'elevator_statistics_report.txt'
    generate_statistics_report(data_frames, report_path)

    # 5. 可视化分析
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)

    create_visualizations(data_frames, results_dir)

    print("\n" + "=" * 60)
    print("分析完成！")
    print(f"📊 统计报告: {report_path}")
    print(f"📈 图表文件: {results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()