# 导入所需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 为了复现结果，设置随机数种子
np.random.seed(42)

# 定义数据集大小
data_size = 1000

# 模拟信贷数据集
# 使用随机数生成年龄、收入、信用评分，以及是否违约的数据
data = {
    'Age': np.random.randint(20, 60, data_size),  # 年龄范围从20到60岁
    'Income': np.random.randint(3000, 10000, data_size),  # 收入范围从3000到10000
    'Credit_Score': np.random.randint(300, 850, data_size),  # 信用评分范围从300到850
    'Default': np.random.choice([0, 1], data_size, p=[0.8, 0.2])  # 是否违约，80%的几率不违约，20%的几率违约
}
df = pd.DataFrame(data)

# 数据可视化部分
# 设置绘图样式
sns.set_style("whitegrid")

# 创建一个2x2的图形布局
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# 使用箱线图展示年龄与违约关系
sns.boxplot(x='Default', y='Age', data=df, ax=ax[0, 0])

# 使用箱线图展示收入与违约关系
sns.boxplot(x='Default', y='Income', data=df, ax=ax[0, 1])

# 使用箱线图展示信用评分与违约关系
sns.boxplot(x='Default', y='Credit_Score', data=df, ax=ax[1, 0])

# 展示违约用户的数量分布
sns.countplot(x='Default', data=df, ax=ax[1, 1])

# 调整布局并展示图形
plt.tight_layout()
plt.show()

# 数据建模部分
# 定义特征和目标变量
X = df[['Age', 'Income', 'Credit_Score']]
y = df['Default']

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用逻辑回归模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 获取模型的分类报告和混淆矩阵
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# 打印分类报告和混淆矩阵
print(report)
print(conf_matrix)


'''
数据分析：利用Python的pandas和matplotlib或seaborn库，对某信贷数据集进行数据清洗、分析和可视化。可以模拟用户的信贷行为数据，分析用户的借贷习惯，信用评级等。
用户分层：使用RFM模型或其他客户分层模型，对用户进行分层管理。
模型建立：利用Python的scikit-learn库，建立一个简单的信贷风险评估模型，如逻辑回归模型，对用户的信贷风险进行评估。
模拟信贷策略制定：基于上述分析和模型，模拟制定信贷策略，如不同信用评级的用户给予不同的借款额度和利率等。
我们已经模拟了一个包含1000条记录的信贷数据集。数据集的字段包括：
    Age: 用户的年龄
    Income: 用户的月收入
    Credit_Score: 用户的信用评分
    Default: 是否违约（0表示没有违约，1表示违约）

接下来，我们将进行以下步骤的分析：
    数据分析和可视化：查看不同特征与违约的关系。
    建立信贷风险评估模型：利用逻辑回归模型进行风险评估。
    年龄与违约关系：违约与非违约的用户年龄分布相似。
    收入与违约关系：违约用户的收入中位数略低于非违约用户。
    信用评分与违约关系：违约用户的信用评分中位数明显低于非违约用户，这意味着信用评分可能是一个强有力的预测因子。
    违约用户的比例：大约20%的用户违约，而80%的用户没有违约。
接下来，我们可以建立一个简单的逻辑回归模型来评估信贷风险。这个模型将基于年龄、收入和信用评分来预测用户是否会违约。
根据逻辑回归模型的分类报告和混淆矩阵，我们可以得出以下结论：
    准确性 (Accuracy)：模型的准确性为83%，意味着模型预测正确的比例为83%。
    查准率 (Precision)：对于非违约用户（标签0），查准率为83%，但对于违约用户（标签1），查准率为0%。这意味着模型没有正确预测出任何一个违约用户。
    查全率 (Recall)：对于非违约用户，查全率为100%，而对于违约用户，查全率为0%。
混淆矩阵的结果显示，模型预测了248个非违约用户，且都预测正确，但对于52个违约用户，模型都预测为非违约，即模型存在偏见，倾向于预测用户为非违约。
需要注意的是，这是一个简单的示例，并不意味着实际应用中的模型会有相同的表现。在实际应用中，我们需要对数据进行更深入的探索和处理，选择合适的模型，并进行调优，以提高模型的性能。

打印的结果：
             precision    recall  f1-score   support

           0       0.83      1.00      0.91       248
           1       0.00      0.00      0.00        52

    accuracy                           0.83       300
   macro avg       0.41      0.50      0.45       300
weighted avg       0.68      0.83      0.75       300

[[248   0]
 [ 52   0]]
'''