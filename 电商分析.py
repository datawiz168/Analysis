import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文字体
myfont = FontProperties(fname="/usr/share/fonts/windows/simhei.ttf")

# 使用随机数种子确保每次运行代码时生成的模拟数据都是相同的
np.random.seed(42)

# 定义数据集的大小
data_size = 1000

# 模拟电商数据
# 随机生成用户ID、年龄、性别、产品类别和购买金额等字段的数据
data = {
    'User_ID': np.arange(1, data_size + 1),  # 生成从1到1000的用户ID
    'Age': np.random.randint(18, 60, data_size),  # 随机生成年龄数据，范围从18到60岁
    'Gender': np.random.choice(['Male', 'Female'], data_size),  # 随机生成性别数据
    'Product_Category': np.random.choice(['Electronics', 'Fashion', 'Home Appliances', 'Books'], data_size),  # 随机生成产品类别数据
    'Purchase_Amount': np.random.randint(50, 500, data_size)  # 随机生成购买金额数据，范围从50到500
}

# 将数据字典转换为DataFrame格式，以便于后续的分析和操作
df = pd.DataFrame(data)

# 用户画像分析
# 获取年龄的基本统计描述（如平均值、中位数、最小值、最大值等）
age_dist = df['Age'].describe()

# 获取性别的分布情况
gender_dist = df['Gender'].value_counts(normalize=True)

# 根据性别统计各产品类别的偏好
product_preference = df.groupby('Gender')['Product_Category'].value_counts(normalize=True)

# 购买习惯分析
# 根据产品类别统计平均购买金额
avg_purchase = df.groupby('Product_Category')['Purchase_Amount'].mean()

# 可视化分析
# 设置图形的大小
plt.figure(figsize=(10, 6))

# 绘制箱线图，展示不同产品类别下的购买金额分布
# 使用myfont设置坐标轴和标题的字体为之前指定的中文字体
sns.boxplot(x='Product_Category', y='Purchase_Amount', data=df).set_title('购买金额分布按产品类别', fontproperties=myfont)
plt.xlabel('产品类别', fontproperties=myfont)
plt.ylabel('购买金额', fontproperties=myfont)

# 显示图形
plt.show()

age_dist, gender_dist, product_preference, avg_purchase
'''
    数据模拟：模拟电商用户购买数据。
    数据探索与可视化：对数据进行基本探索并可视化。
    用户画像分析：基于模拟数据构建用户画像。
    购买习惯分析：分析用户的购买习惯。
'''