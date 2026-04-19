#陈硕勋-25343004-第三次人工智能编程作业
## 1.任务拆解与AI写协作策略
我先按任务文件中的六个任务作为六个大目标，再逐一拆解每个目标，细化到一个个小的任务，例如读取并打印数据、计算搭乘站点数并创建新列ride_stops、排查出各列缺失值数量并打印等等。
任务一：
用pandas读取数据并打印前五行和基本信息
提取该列中“小时”字段，新增hour列
计算搭乘站点数并创建新列ride_stops
排查出各列缺失值数量并打印，若存在缺失值，删除
任务二：
使用numpy。统计早晚刷卡量，使用布尔索引
使用matplotlib。绘制一张24小时刷卡量柱状图。
保存这张图，名字为hour_distribution.png，dpi=150
任务三：
定义一个函数，读取df, route_col='线路号', stops_col='ride_stops'，计算个线路乘客的平均搭乘站点数和标准差。
打印结果输出前十行
用seaborn对返回的结果可视化
任务四：
不指定高峰时段，统计全天份小时统计结果，自动找出刷卡量最大的那个小时作为高峰小时并输出
规定输出格式
任务五：
筛选路线1101至1120之间的所有记录，在根目录下生成 路线驾驶员信息 文件夹
输出该线路中所有出现过的（车辆编号 → 驾驶员编号）对应关系（去重），写入以线路号命名的 txt 文件（如 1101.txt）
打印20个文件的生成路径
任务六：
做一个排名统计，找出服务人次最多的TOP10司机、TOP10线路、TOP10上车点和TOP10车辆，打印结果，可以不用函数
制作热力图，构造4*10热力图，行是4个维度，列是各维度top10实体，以服务人次为色值

## 2.核心 Prompt 迭代记录
初代prompt：
import pandas as pd
import numpy as np

def calculate_phf(df):
    """
    计算高峰小时系数 (PHF5 和 PHF15)
    """
    # 确保时间列是 datetime 类型
    if not np.issubdtype(df['刷卡时间'].dtype, np.datetime64):
        df['刷卡时间'] = pd.to_datetime(df['刷卡时间'])

    # 1. 统计全天每小时刷卡量
    # 按小时分组统计数量
    hourly_counts = df.groupby(df['刷卡时间'].dt.hour).size()
    
    # 2. 找出高峰小时 (刷卡量最大的那个小时)
    peak_hour = hourly_counts.idxmax()
    peak_hour_volume = hourly_counts.max()
    
    print(f"📊 全天刷卡量统计完成")
    print(f"🚀 高峰小时: {peak_hour}点")
    print(f"🔢 高峰小时刷卡量: {peak_hour_volume}")

    # 3. 提取高峰小时的数据子集
    peak_hour_data = df[df['刷卡时间'].dt.hour == peak_hour]
    
    # 4. 统计每 5 分钟和每 15 分钟的刷卡量
    # 利用 dt.floor 将时间向下取整到最近的 5分钟 或 15分钟
    # 例如: 07:08 -> 07:05, 07:14 -> 07:00
    
    # 统计每5分钟的量
    counts_5min = peak_hour_data.groupby(peak_hour_data['刷卡时间'].dt.floor('5min')).size()
    max_5min_volume = counts_5min.max()
    
    # 统计每15分钟的量
    counts_15min = peak_hour_data.groupby(peak_hour_data['刷卡时间'].dt.floor('15min')).size()
    max_15min_volume = counts_15min.max()

    # 5. 计算 PHF
    # PHF5 = 高峰小时刷卡量 / (12 * 高峰小时内最大5分钟刷卡量)
    phf5 = peak_hour_volume / (12 * max_5min_volume)
    
    # PHF15 = 高峰小时刷卡量 / (4 * 高峰小时内最大15分钟刷卡量)
    phf15 = peak_hour_volume / (4 * max_15min_volume)

    return phf5, phf15

    # --- 调用示例 ---

    # 假设 df 是你的原始数据，且有一列叫 '刷卡时间'
    # phf5_result, phf15_result = calculate_phf(df)

    # 打印结果 (保留4位小数)
    # print(f"PHF5 计算结果: {phf5_result:.4f}")
    # print(f"PHF15 计算结果: {phf15_result:.4f}")
AI的问题：与要求输出格式不符，没有用ICData.csv文件中的列名
优化后的prompt：
import pandas as pd
import numpy as np

def calculate_phf_formatted(df):
    """
    计算并打印指定格式的 PHF 报告
    """
    # 确保时间列是 datetime 类型
    if not np.issubdtype(df['交易时间'].dtype, np.datetime64):
        df['刷卡时间'] = pd.to_datetime(df['交易时间'])

    # 1. 统计全天每小时刷卡量
    hourly_counts = df.groupby(df['交易时间'].dt.hour).size()
    
    # 2. 找出高峰小时
    peak_hour = hourly_counts.idxmax()
    peak_hour_volume = hourly_counts.max()
    
    # 3. 提取高峰小时的数据子集
    peak_hour_data = df[df['交易时间'].dt.hour == peak_hour]
    
    # 4. 统计每 5 分钟和每 15 分钟的刷卡量
    # 使用 dt.floor 向下取整
    counts_5min = peak_hour_data.groupby(peak_hour_data['交易时间'].dt.floor('5min')).size()
    max_5min_volume = counts_5min.max()
    # 找到最大5分钟对应的起始时间
    max_5min_start_time = counts_5min.idxmax()
    
    counts_15min = peak_hour_data.groupby(peak_hour_data['交易时间'].dt.floor('15min')).size()
    max_15min_volume = counts_15min.max()
    # 找到最大15分钟对应的起始时间
    max_15min_start_time = counts_15min.idxmax()

    # 5. 计算 PHF
    phf5 = peak_hour_volume / (12 * max_5min_volume)
    phf15 = peak_hour_volume / (4 * max_15min_volume)

    # 6. 格式化时间字符串 (HH:MM)
    # 高峰小时结束时间 = 开始时间 + 1小时
    start_h = int(peak_hour)
    end_h = start_h + 1
    
    # 最大5分钟结束时间
    max_5_end = max_5min_start_time + pd.Timedelta(minutes=5)
    
    # 最大15分钟结束时间
    max_15_end = max_15min_start_time + pd.Timedelta(minutes=15)

    # 7. 按指定格式打印
    print(f"高峰小时：{start_h:02d}:00 ~ {end_h:02d}:00，刷卡量：{peak_hour_volume} 次")
    print(f"最大5分钟刷卡量（{max_5min_start_time.strftime('%H:%M')}~{max_5_end.strftime('%H:%M')}）：{max_5min_volume} 次")
    print(f"PHF5  = {peak_hour_volume} / (12 × {max_5min_volume}) = {phf5:.4f}")
    print(f"最大15分钟刷卡量（{max_15min_start_time.strftime('%H:%M')}~{max_15_end.strftime('%H:%M')}）：{max_15min_volume} 次")
    print(f"PHF15 = {peak_hour_volume} / ( 4 × {max_15min_volume}) = {phf15:.4f}")

    # --- 调用示例 ---

    # 假设 df 是你的原始数据，且有一列叫 '刷卡时间'
    # calculate_phf_formatted(df)

## 3. Debug 与异常处理记录
报错类型：
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/pandas/core/indexes/base.py", line 3641, in get_loc
    return self._engine.get_loc(casted_key)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "pandas/_libs/index.pyx", line 168, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 197, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7668, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7676, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: '交易时间'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/astronaut_non/Desktop/课程/程序设计/pc python/week_6/homework/demo.py", line 14, in <module>
    df['交易时间'] = pd.to_datetime(df['交易时间'], errors='coerce')
                                    ~~^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/pandas/core/frame.py", line 4378, in __getitem__
    indexer = self.columns.get_loc(key)
  File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/pandas/core/indexes/base.py", line 3648, in get_loc
    raise KeyError(key) from err
KeyError: '交易时间'
解决过程：
返丢给AI，发现是未识别出表头的原因，于是通过print(df.columns)检查表头，发现所有表头并成了一个，没有分隔开，于是更改了分隔符，将“\t”改回“,”，成功切割表头，因此python才能识别'交易时间'

## 4.人工代码审查 (Code Review)
    #读取数据
    df = pd.read_csv('ICData.csv')
    #打印前五行数据
    print("--- 数据预览 (前5行) ---")
    print(df.head())
    #打印数据基本信息
    print("\n--- 数据基本信息 ---")
    print(df.info())
    #交易时间 列的类型更改为datatime
    df['交易时间'] = pd.to_datetime(df['交易时间'], errors='coerce')
    # 提取“交易时间”列的小时部分，并赋值给新列 'hour'
    df['hour'] = df['交易时间'].dt.hour
    #新增搭乘站点数列
    df['上车站点'] = pd.to_numeric(df['上车站点'], errors='coerce')
    df['下车站点'] = pd.to_numeric(df['下车站点'], errors='coerce')
    df['ride_stops'] = (df['上车站点'] - df['下车站点']).abs()
    #找出异常
    abnormal_rows = df[df['ride_stops'] == 0]
    deleted_count = len(abnormal_rows)
    df = df[df['ride_stops'] != 0].reset_index(drop=True)
    print(f"🗑️ 已删除异常行数: {deleted_count}")
    #找出缺失值
    missing_values = df.isnull().sum()
    print("🔍 各列缺失值数量统计：")
    print(missing_values)
    #删除
    total_missing = missing_values.sum()
    if total_missing > 0:
        print(f"\n⚠️ 发现总计 {total_missing} 个缺失值，正在删除包含缺失值的行...")
        rows_before = len(df)
        # 删除任何包含缺失值的行
        # how='any' 表示只要有一个缺失就删，axis=0 表示按行删
        df = df.dropna(how='any', axis=0).reset_index(drop=True)
        # 计算删除的行数
        rows_deleted = rows_before - len(df)
        print(f"✅ 清洗完成，共删除 {rows_deleted} 行数据。")
    else:
        print("\n✅ 数据完整，未发现缺失值。")