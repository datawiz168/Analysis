import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 步骤1：生成模拟的电商平台汽车销售数据
np.random.seed(42)
num_samples = 1000

# 创建模拟数据集
data = {
    "Car_Brand": np.random.choice(["BrandA", "BrandB", "BrandC", "BrandD"], num_samples),
    "Car_Age": np.random.randint(1, 10, num_samples),
    "Mileage": np.random.randint(1000, 100000, num_samples),
    "Sale_Price": np.random.randint(5000, 50000, num_samples)
}

df = pd.DataFrame(data)

# 步骤2：使用pandas和numpy进行数据分析

# 获取基本统计信息
basic_stats = df.describe()

print("基本统计信息:")
print(basic_stats)

# 按品牌计算平均售价
avg_price_per_brand = df.groupby("Car_Brand")["Sale_Price"].mean()

print("\n按品牌划分的平均销售价格:")
print(avg_price_per_brand)

# 步骤3：使用机器学习模型基于汽车的车龄和里程预测其售价

X = df[["Car_Age", "Mileage"]]  # 选择特征
y = df["Sale_Price"]  # 选择目标变量

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 计算模型的均方误差
mse = mean_squared_error(y_test, y_pred)

print("\n机器学习模型的均方误差:")
print(mse)
