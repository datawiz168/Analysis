# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 设置随机数种子以确保每次运行结果的一致性
np.random.seed(42)

# 定义数据集大小
data_size = 1000

# 模拟游戏数据
# 随机生成玩家ID、留存天数、付费金额、活动参与次数等字段的数据
data = {
    'Player_ID': np.arange(1, data_size + 1),  # 生成从1到1000的用户ID
    'Retention_Days': np.random.randint(1, 365, data_size),  # 使用整数随机数模拟留存天数
    'Payment': np.random.randint(0, 100, data_size),  # 使用整数随机数模拟付费金额
    'Event_Participation': np.random.poisson(2, data_size)  # 使用泊松分布模拟活动参与次数
}

# 将模拟的数据转化为DataFrame格式，方便后续分析
df = pd.DataFrame(data)

# 付费分析：计算平均付费金额
avg_payment = df['Payment'].mean()

# 活动效果分析：计算平均活动参与次数
avg_event_participation = df['Event_Participation'].mean()

# 精细化运营：使用KMeans进行用户分层
# 使用KMeans聚类算法将玩家分为3个群体/层次
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(df[['Retention_Days', 'Payment', 'Event_Participation']])

# 数据可视化
# 使用散点图展示留存天数与付费金额，并根据用户分层结果标记颜色
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Retention_Days', y='Payment', hue='Cluster')
plt.title('基于留存和付费的用户分层')  # 设置图的标题
plt.show()

# 输出平均付费金额和平均活动参与次数的结果
avg_payment, avg_event_participation


'''
    游戏数据分析：例如用户留存、付费分析、活动效果。
    数据处理与精细化运营：模拟数据预处理、用户分层、以及基于数据的运营建议。
    BI系统与数据应用平台：虽然在此脚本中我们不能完整展示BI系统的搭建，但可以使用Python模拟数据分析模型的构建。
这段代码首先模拟了游戏玩家的基本数据，然后进行了付费和活动参与分析，接着使用了KMeans聚类算法进行用户分层，并通过可视化展示了结果。
'''