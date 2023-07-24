# 导入所需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns

# 设置中文字体，确保中文可以正常显示
myfont = FontProperties(fname="/usr/share/fonts/windows/simhei.ttf")

# 设置随机数种子，保证每次运行时生成的模拟数据一致
np.random.seed(42)

# 模拟用户数据
# -------------------------------------------------------------
# 为了展示社交数据分析的能力，我们首先模拟一些用户数据。
# 这包括用户的活跃度、7日留存率、发布的内容数量、与好友的互动次数以及用户的标签。
# -------------------------------------------------------------

user_data = {
    '用户ID': range(1, 101),
    '活跃度': np.random.randint(1, 100, 100),  # 模拟用户的活跃度
    '7日留存': np.random.choice([0, 1], 100, p=[0.5, 0.5]),  # 模拟用户7日后是否还留存
    '发布内容数量': np.random.poisson(5, 100),  # 假设用户发布内容数量服从泊松分布
    '好友互动次数': np.random.poisson(10, 100),  # 假设用户与好友的互动次数服从泊松分布
    '用户标签': np.random.choice(['新用户', '老用户', '内容创作者'], 100)  # 随机为用户打上标签
}

# 根据上述模拟数据创建DataFrame数据框
df = pd.DataFrame(user_data)

# 数据分析与可视化
# -------------------------------------------------------------
# 利用模拟的数据进行数据分析，并生成可视化结果。
# -------------------------------------------------------------

# 画出活跃度与7日留存的关系的箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['7日留存'], y=df['活跃度'], palette="pastel")
plt.title('活跃度与7日留存的关系', fontproperties=myfont)
plt.xlabel('7日留存', fontproperties=myfont)
plt.ylabel('活跃度', fontproperties=myfont)
plt.xticks([0, 1], ['未留存', '留存'], fontproperties=myfont)
plt.show()

# 画出发布内容数量与好友互动次数的关系的散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['发布内容数量'], y=df['好友互动次数'], hue=df['用户标签'], palette="pastel")
plt.title('发布内容数量与好友互动次数的关系', fontproperties=myfont)
plt.xlabel('发布内容数量', fontproperties=myfont)
plt.ylabel('好友互动次数', fontproperties=myfont)
plt.legend(prop=myfont)
plt.show()

# 画出用户标签与活跃度/留存的关系的箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['用户标签'], y=df['活跃度'], hue=df['7日留存'], palette="pastel")
plt.title('用户标签与活跃度及留存的关系', fontproperties=myfont)
plt.xlabel('用户标签', fontproperties=myfont)
plt.ylabel('活跃度', fontproperties=myfont)
plt.xticks(fontproperties=myfont)  # 修复方框问题
legend = plt.legend(title="7日留存", prop=myfont, loc='upper left')
legend.get_title().set_fontproperties(myfont)  # 设置图例标题的字体
plt.show()


'''
简介与说明：
本代码旨在模拟社交数据并进行基础的数据分析和可视化，以展示在社交数据分析方面的能力。
具体来说，我们模拟了一些假设的用户数据，包括活跃度、7日留存率、发布内容数量、与好友的互动次数以及用户的标签。
通过这些数据，我们生成了三个可视化结果：活跃度与7日留存的关系、发布内容数量与好友互动次数的关系、以及用户标签与活跃度/留存的关系。
这些结果为我们提供了对模拟数据的深入理解，帮助我们从数据中发现潜在的规律和趋势。
'''