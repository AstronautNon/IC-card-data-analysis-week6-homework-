import pandas as pd
#任务1
#读取数据
df = pd.read_csv('ICData.csv', sep='\t')
#打印前五行数据
print("--- 数据预览 (前5行) ---")
print(df.head())
#打印数据基本信息
print("\n--- 数据基本信息 ---")
print(df.info())
#交易时间 列的类型更改为datatime
df['交易时间'] = pd.to_datetime(df['交易时间'], errors='coerce')
