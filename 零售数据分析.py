import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 模拟零售数据
np.random.seed(42)
data_size = 1000

# 随机生成产品类别、销售额、成本、月份等数据
data = {
    'Product_Category': np.random.choice(['Fresh Produce', 'Standard Goods', 'Processed Goods'], data_size, p=[0.4, 0.3, 0.3]),
    'Sales': np.random.randint(50, 500, data_size),
    'Cost': np.random.randint(20, 400, data_size),
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr'], data_size)
}

df = pd.DataFrame(data)

# 计算利润
df['Profit'] = df['Sales'] - df['Cost']

# 数据运营与经营分析
monthly_sales = df.groupby('Month')['Sales'].sum()
monthly_profit = df.groupby('Month')['Profit'].sum()

# 品类规划与分析
category_sales = df.groupby('Product_Category')['Sales'].sum()
category_profit = df.groupby('Product_Category')['Profit'].sum()

# 可视化
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

sns.barplot(x=monthly_sales.index, y=monthly_sales.values, ax=ax[0])
ax[0].set_title('Monthly Sales Analysis')
ax[0].set_ylabel('Sales')

sns.barplot(x=category_sales.index, y=category_sales.values, ax=ax[1])
ax[1].set_title('Product Category Sales Analysis')
ax[1].set_ylabel('Sales')

plt.tight_layout()
plt.show()

# 这段代码首先模拟了零售业务的销售数据，然后进行了按月份和产品类别的销售分析，并通过可视化展示了结果。