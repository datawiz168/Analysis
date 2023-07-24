# 导入所需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 为了显示中文，设置字体属性
myfont = FontProperties(fname="/usr/share/fonts/windows/simhei.ttf")

# 为了使结果可重复，设置随机数种子
np.random.seed(42)
data_size = 1000

# 以下代码用于模拟生成会员数据
# 会员ID: 从1到1000
# 会员等级: 随机分配为Silver, Gold, Platinum
# 购买次数: 泊松分布模拟
# 平均消费金额: 正态分布模拟
# 最后购买日期: 在过去的365天内随机选择
data = {
    'Member_ID': np.arange(1, data_size + 1),
    'Member_Level': np.random.choice(['Silver', 'Gold', 'Platinum'], data_size),
    'Purchase_Times': np.random.poisson(5, data_size),
    'Avg_Spending': np.random.normal(300, 50, data_size),
    'Last_Purchase_Days': np.random.randint(1, 365, data_size)
}

# 将字典数据转换为DataFrame
df = pd.DataFrame(data)

# 对Platinum会员的购买次数进行平均计算，作为一个KPI指标
platinum_avg_purchase = df[df['Member_Level'] == 'Platinum']['Purchase_Times'].mean()

# 使用seaborn库进行数据可视化，展示不同会员等级的平均消费金额
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Member_Level', y='Avg_Spending')
plt.title('会员等级 vs 平均消费金额', fontproperties=myfont)
plt.xlabel('会员等级', fontproperties=myfont)
plt.ylabel('平均消费金额', fontproperties=myfont)
plt.show()

# 输出Platinum会员的平均购买次数
platinum_avg_purchase



'''
会员业务数据分析：特别是关于人、货、场的数据洞察。
KPI指标制定与监测：制定KPI并进行持续监测，及时识别问题。
数据分析体系制定：特别是和商品及渠道部门合作，建立分析体系。
数据系统与报表模块设计：理解业务需求，设计相应的数据报表。
会员运营数据支持：为会员运营、活动、推广提供数据支
这个脚本模拟了不同会员等级的平均消费金额，并展示了Platinum会员的平均购买次数。

'''