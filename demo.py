#库导入区
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#函数定义区
def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    # 1. 分组并聚合计算
    # 对指定的线路列分组，对站点列计算均值和标准差
    stats = df.groupby(route_col)[stops_col].agg(
        mean_stops='mean',  # 计算平均值
        std_stops='std'  # 计算标准差
    ).reset_index()  # 将分组后的索引还原为普通列

    # 2. 排序
    # 按平均搭乘站点数降序排列
    stats = stats.sort_values(by='mean_stops', ascending=False).reset_index(drop=True)

    return stats


def calculate_phf(df):
    """
    计算高峰小时系数 (PHF5 和 PHF15)
    """
    # 确保时间列是 datetime 类型
    if not np.issubdtype(df['交易时间'].dtype, np.datetime64):
        df['交易时间'] = pd.to_datetime(df['交易时间'])

    # 1. 统计全天每小时刷卡量
    # 按小时分组统计数量
    hourly_counts = df.groupby(df['交易时间'].dt.hour).size()

    # 2. 找出高峰小时 (刷卡量最大的那个小时)
    peak_hour = hourly_counts.idxmax()
    peak_hour_volume = hourly_counts.max()

    print(f"📊 全天刷卡量统计完成")
    print(f"🚀 高峰小时: {peak_hour}点")
    print(f"🔢 高峰小时刷卡量: {peak_hour_volume}")

    # 3. 提取高峰小时的数据子集
    peak_hour_data = df[df['交易时间'].dt.hour == peak_hour]

    # 4. 统计每 5 分钟和每 15 分钟的刷卡量
    # 利用 dt.floor 将时间向下取整到最近的 5分钟 或 15分钟
    # 例如: 07:08 -> 07:05, 07:14 -> 07:00

    # 统计每5分钟的量
    counts_5min = peak_hour_data.groupby(peak_hour_data['交易时间'].dt.floor('5min')).size()
    max_5min_volume = counts_5min.max()

    # 统计每15分钟的量
    counts_15min = peak_hour_data.groupby(peak_hour_data['交易时间'].dt.floor('15min')).size()
    max_15min_volume = counts_15min.max()

    # 5. 计算 PHF
    # PHF5 = 高峰小时刷卡量 / (12 * 高峰小时内最大5分钟刷卡量)
    phf5 = peak_hour_volume / (12 * max_5min_volume)

    # PHF15 = 高峰小时刷卡量 / (4 * 高峰小时内最大15分钟刷卡量)
    phf15 = peak_hour_volume / (4 * max_15min_volume)

    return phf5, phf15

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
#绘图
# 统计每个小时的刷卡量
# value_counts() 统计频次，sort_index() 按小时排序，reindex 确保 0-23 点都有数据（没有的补0）
hourly_counts = df['hour'].value_counts().sort_index().reindex(range(24), fill_value=0)
# 提取 x轴 (小时) 和 y轴 (数量)
x_hours = hourly_counts.index
y_counts = hourly_counts.values
# 定义高亮时段的掩码
# 早峰前: 0, 1, ..., 6
morning_mask = x_hours < 7
# 深夜: 22, 23
night_mask = x_hours >= 22
# 普通时段: 其他
normal_mask = ~(morning_mask | night_mask)
#设置绘图风格和中文支持
plt.style.use('seaborn-v0_8-whitegrid') # 使用简洁的网格风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
#创建画布
plt.figure(figsize=(14, 7))
#分段绘制柱状图
# 第一步：先画普通时段（灰色）
plt.bar(x_hours[normal_mask], y_counts[normal_mask], color='#d3d3d3', label='普通时段')
# 第二步：画早高峰前（蓝色），覆盖在上方
plt.bar(x_hours[morning_mask], y_counts[morning_mask], color='#4c72b0', label='早高峰前 (0-6点)')
# 第三步：画深夜时段（红色），覆盖在上方
plt.bar(x_hours[night_mask], y_counts[night_mask], color='#c44e52', label='深夜 (22-23点)')
#图表美化
plt.title('24小时刷卡量分布可视化', fontsize=18, pad=20)
plt.xlabel('小时', fontsize=12)
plt.ylabel('刷卡量 (次)', fontsize=12)
#设置 x 轴刻度：0 到 23，步长为 2
plt.xticks(range(0, 24, 2))
#添加水平网格线 (axis='y' 表示只显示水平线)
plt.grid(axis='y', linestyle='--', alpha=0.7)
#显示图例
plt.legend()
#自动调整布局，防止标签被遮挡
plt.tight_layout()
#保存图像
#plt.savefig('hour_distribution.png', dpi=150, bbox_inches='tight')
print("✅ 图表已成功保存为 'hour_distribution.png'")
#显示图表
plt.show()

#任务三
#调用函数
result_df = analyze_route_stops(df)
#打印前十行
print("📊 各线路平均搭乘站点数统计 (前10名)：")
print(result_df.head(10))
#可视化
plot_data = result_df.head(15)
plt.figure(figsize=(12, 8))
sns.barplot(
    data=plot_data,
    x='线路号',
    y='mean_stops',
    errorbar=('sd', 0),
    capsize=0.3,
    dodge=False,
    palette='viridis'
)
plt.title('各线路平均搭乘站点数 Top 10 (带标准差)', fontsize=16, pad=20)
plt.xlabel('线路号', fontsize=12)
plt.ylabel('平均搭乘站点数', fontsize=12)
# 设置 y 轴范围从 0 开始，避免截断柱子
plt.ylim(bottom=0)
# 添加网格线，方便读数
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 保存图像
#plt.savefig('route_stops.png', dpi=150, bbox_inches='tight')
print("✅ 图像已保存为 route_stops.png")
# 显示图像
plt.show()

#任务四
phf5_result, phf15_result = calculate_phf(df)