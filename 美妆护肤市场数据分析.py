# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文字体，确保中文可以正常显示
myfont = FontProperties(fname="/usr/share/fonts/windows/simhei.ttf")

# 模拟美妆护肤市场规模数据
categories = ['面霜', '口红', '眼影', '粉底', '卸妆油']
market_size = [5000, 8000, 3000, 4000, 2500]

# 模拟消费者偏好数据
preferences = ['自然', '夸张', '经典', '现代', '复古']
preference_percentage = [0.4, 0.1, 0.25, 0.2, 0.05]

# 模拟内部产品销售数据
sales_data = {
    '产品': ['面霜A', '面霜B', '口红A', '口红B', '眼影A'],
    '销售量': [1000, 800, 1500, 1400, 600],
    '评分': [4.5, 4.2, 4.8, 4.1, 4.7]
}
df_sales = pd.DataFrame(sales_data)

# 绘制美妆护肤市场规模柱状图
plt.figure(figsize=(10, 5))
sns.barplot(x=categories, y=market_size, palette="pastel")
plt.title('美妆护肤市场规模', fontproperties=myfont)
plt.xlabel('产品类别', fontproperties=myfont)
plt.ylabel('市场规模 (单位：万元)', fontproperties=myfont)
plt.xticks(fontproperties=myfont)  # 设置x轴标签的字体属性为中文
plt.show()

# 绘制消费者偏好走向柱状图
plt.figure(figsize=(10, 5))
sns.barplot(x=preferences, y=preference_percentage, palette="pastel")
plt.title('消费者偏好走向', fontproperties=myfont)
plt.xlabel('偏好类型', fontproperties=myfont)
plt.ylabel('百分比', fontproperties=myfont)
plt.xticks(fontproperties=myfont)  # 设置x轴标签的字体属性为中文
plt.show()

# 绘制内部产品销售数据柱状图
plt.figure(figsize=(10, 5))
sns.barplot(x=df_sales['产品'], y=df_sales['销售量'], palette="pastel")
plt.title('内部产品销售数据', fontproperties=myfont)
plt.xlabel('产品', fontproperties=myfont)
plt.ylabel('销售量', fontproperties=myfont)
plt.xticks(fontproperties=myfont)  # 设置x轴标签的字体属性为中文
plt.show()
