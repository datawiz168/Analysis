import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置字体属性
myfont = FontProperties(fname="/usr/share/fonts/windows/simhei.ttf")

# 模拟数据
np.random.seed(42)
data_size = 1000

# 生成随机的业务数据：产品ID, 产品成本, 销售量
data = {
    'Product_ID': np.arange(1, data_size + 1),
    'Product_Cost': np.random.normal(50, 10, data_size),
    'Sales_Quantity': np.random.poisson(20, data_size)
}

df = pd.DataFrame(data)

# 成本优化：识别高成本产品
high_cost_products = df[df['Product_Cost'] > df['Product_Cost'].quantile(0.9)]

# 需求增长：识别低销售量产品
low_sales_products = df[df['Sales_Quantity'] < df['Sales_Quantity'].quantile(0.1)]

# 数据可视化：成本与销售量
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Product_Cost', y='Sales_Quantity')
plt.title('产品成本 vs 销售量', fontproperties=myfont)  # 使用设置的字体
plt.xlabel('产品成本', fontproperties=myfont)  # 使用设置的字体
plt.ylabel('销售量', fontproperties=myfont)  # 使用设置的字体
plt.show()

high_cost_products, low_sales_products
