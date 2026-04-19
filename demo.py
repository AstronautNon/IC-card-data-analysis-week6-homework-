#库导入区
import pandas as pd
import numpy as np

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

