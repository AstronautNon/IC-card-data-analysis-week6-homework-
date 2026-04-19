#库导入区
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#任务1
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

#任务二
#统计早高峰前和深夜时段的刷卡量
#提取相关列转换为 NumPy 数组
swipe_type = df['刷卡类型'].values
hours = df['hour'].values
#计算全天总刷卡量
total_count = len(df)
#使用布尔索引进行筛选
# 逻辑：(刷卡类型为0) 与 (小时 < 7)
morning_mask = (swipe_type == 0) & (hours < 7)
morning_count = np.sum(morning_mask) # 统计 True 的数量
# 逻辑：(刷卡类型为0) 与 (小时 >= 22)
night_mask = (swipe_type == 0) & (hours >= 22)
night_count = np.sum(night_mask) # 统计 True 的数量
#计算百分比
if total_count > 0:
    morning_pct = (morning_count / total_count) * 100
    night_pct = (night_count / total_count) * 100
else:
    morning_pct = night_pct = 0
#打印结果
print("📊 刷卡类型为0的统计分析 (基于NumPy):")
print("-" * 30)
print(f"全天总刷卡量: {total_count}")
print(f"早高峰前 (hour < 7): {morning_count} 次 (占比: {morning_pct:.2f}%)")
print(f"深夜时段 (hour >= 22): {night_count} 次 (占比: {night_pct:.2f}%)")
