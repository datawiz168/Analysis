import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 模拟游戏数据
np.random.seed(42)
data_size = 1000

# 随机生成玩家ID、留存天数、是否流失、付费金额等数据
data = {
    'Player_ID': np.arange(1, data_size + 1),
    'Retention_Days': np.random.randint(1, 365, data_size),
    'Churned': np.random.choice([0, 1], data_size, p=[0.7, 0.3]),
    'Payment': np.random.randint(0, 100, data_size)
}

df = pd.DataFrame(data)

# 留存分析
retention_avg = df['Retention_Days'].mean()

# 付费分析
paying_users = df[df['Payment'] > 0].shape[0]

# 预流失模型
X = df[['Retention_Days', 'Payment']]
y = df['Churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# 数据可视化
plt.figure(figsize=(12, 6))
sns.histplot(df[df['Churned'] == 0]['Retention_Days'], label='Not Churned', kde=True)
sns.histplot(df[df['Churned'] == 1]['Retention_Days'], label='Churned', kde=True, color='red')
plt.title('Retention Days Distribution by Churn Status')
plt.legend()
plt.show()

retention_avg, paying_users, accuracy
'''
    游戏数据分析：例如用户留存、流失、付费分析。
    数据拉取与处理：模拟如何从数据库获取数据。
    预测模型：例如预流失模型。
    数据可视化与报告制作：使用Python制作数据图表。

这段代码首先模拟了游戏玩家的基本数据，然后进行了留存和付费分析，接着使用了随机森林模型预测玩家流失，并最后通过可视化展示了玩家的留存天数分布。


'''