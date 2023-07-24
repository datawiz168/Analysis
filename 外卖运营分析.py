# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib.font_manager import FontProperties

# 设置中文字体，确保图表中的中文可以正常显示
myfont = FontProperties(fname="/usr/share/fonts/windows/simhei.ttf")

# 创建模拟数据
# 这里我们假设有5个店铺，每个店铺都有关于它的多个指标
data = {
    '店铺名': ['店铺A', '店铺B', '店铺C', '店铺D', '店铺E'],
    '排名因子': np.random.randint(60, 100, 5),
    '店铺分': np.random.randint(60, 100, 5),
    '门店星级': np.random.randint(3, 6, 5),
    '流量指标': np.random.randint(500, 1500, 5),
    '毛利率': np.random.uniform(0.1, 0.3, 5),
    '利润率': np.random.uniform(0.05, 0.2, 5),
    '边际贡献': np.random.randint(1000, 5000, 5)
}
df = pd.DataFrame(data)  # 将字典转换为pandas DataFrame

# 数据清洗
# 对数值型特征进行归一化处理，使其在0-1之间，这样可以更好地在图表中进行比较
scaler = MinMaxScaler()
df[['排名因子', '店铺分', '流量指标', '毛利率', '利润率', '边际贡献']] = scaler.fit_transform(df[['排名因子', '店铺分', '流量指标', '毛利率', '利润率', '边际贡献']])

# 数据分析 & 可视化
# 利润率与毛利率的关系散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['毛利率'], y=df['利润率'], hue=df['店铺名'], s=100)  # 使用seaborn的scatterplot函数绘制散点图
plt.title('毛利率与利润率的关系', fontproperties=myfont)  # 设置图表标题
plt.xlabel('毛利率', fontproperties=myfont)  # 设置x轴标签
plt.ylabel('利润率', fontproperties=myfont)  # 设置y轴标签
plt.legend(prop=myfont)  # 设置图例，并确保图例中的中文可以正常显示
plt.grid(True)  # 显示网格线
plt.show()  # 显示图表

# 边际贡献与流量指标的关系柱状图
plt.figure(figsize=(10, 6))
sns.barplot(x=df['店铺名'], y=df['边际贡献'], hue=df['门店星级'], palette="pastel")  # 使用seaborn的barplot函数绘制柱状图
plt.title('店铺边际贡献与流量指标的关系', fontproperties=myfont)  # 设置图表标题
plt.xlabel('店铺名', fontproperties=myfont)  # 设置x轴标签
plt.ylabel('边际贡献', fontproperties=myfont)  # 设置y轴标签
plt.xticks(fontproperties=myfont)  # 确保x轴的中文可以正常显示
legend = plt.legend(title='门店星级', prop=myfont, loc='upper left')  # 设置图例，并确保图例中的中文可以正常显示
legend.get_title().set_fontproperties(myfont)  # 确保图例标题的中文可以正常显示
plt.show()  # 显示图表

