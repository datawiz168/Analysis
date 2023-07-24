# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# 设置随机数种子以确保每次运行结果的一致性
np.random.seed(42)

# 定义数据集大小
data_size = 1000

# 模拟电商数据
# 随机生成用户ID、购买次数、购买金额、产品类别等字段的数据
data = {
    'User_ID': np.arange(1, data_size + 1),  # 生成从1到1000的用户ID
    'Purchase_Times': np.random.poisson(2, data_size),  # 使用泊松分布模拟购买次数
    'Purchase_Amount': np.random.exponential(200, data_size),  # 使用指数分布模拟购买金额
    'Product_Category': np.random.choice(['Electronics', 'Fashion', 'Home Appliances'], data_size)  # 随机选择产品类别
}

# 将模拟的数据转化为DataFrame格式，方便后续分析
df = pd.DataFrame(data)

# 数据异常监控
# 使用孤立森林算法检测异常值，将检测到的异常值标记为-1，正常值标记为1
clf = IsolationForest(contamination=0.05)  # contamination参数设置异常值的比例
df['Anomaly'] = clf.fit_predict(df[['Purchase_Times', 'Purchase_Amount']])

# 数据探索与机会挖掘
# 对产品类别进行分组，并计算每个类别的总购买金额
category_sales = df.groupby('Product_Category')['Purchase_Amount'].sum()

# 找出购买次数在90%分位数以上的频繁购买者
frequent_buyers = df[df['Purchase_Times'] > df['Purchase_Times'].quantile(0.9)]

# 数据可视化
# 使用散点图展示购买次数与购买金额的关系，并根据异常检测的结果标记颜色
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Purchase_Times', y='Purchase_Amount', hue='Anomaly', palette=['red', 'green'])
plt.title('购买分析与异常检测')  # 设置图的标题
plt.show()

# 输出类别销售总额和频繁购买者的信息
category_sales, frequent_buyers



'''
    电商数据分析：例如用户留存、转化率和平均订单价值。
    数据异常监控：模拟数据异常检测。
    数据探索与机会挖掘：例如购买频率分析、产品偏好分析。
    使用SQL和Python：虽然在此脚本中我们不能直接使用SQL，但可以用Python模拟数据查询和处理。

这段代码首先模拟了电商购买数据，然后使用孤立森林算法进行了异常检测。接着，进行了简单的数据探索和机会挖掘，并使用可视化展示了购买分析与异常点。

'''