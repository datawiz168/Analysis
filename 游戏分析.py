# 场景描述：
# 你是一家大型在线多人角色扮演游戏（MMORPG）的数据分析师。游戏中的玩家可以完成任务、升级、购买虚拟商品等。
# 最近，游戏公司推出了一个新的虚拟物品，并希望了解这个物品的销售表现、对玩家留存的影响等。

# 分析目标：
# 1. 分析新虚拟物品的销售趋势。
# 2. 了解购买新虚拟物品的玩家与未购买的玩家在活跃度、任务完成情况等方面的差异。
# 3. 预测未来一周的物品销售情况。

# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# 设置字体，确保中文正常显示
myfont = FontProperties(fname="/usr/share/fonts/windows/simhei.ttf")

# 生成模拟的游戏数据
num_players = 1000
num_days = 30
dates = [datetime.now() - timedelta(days=i) for i in range(num_days)]
player_ids = range(1, num_players + 1)

# 使用随机数据模拟玩家的行为
data = {
    "Date": np.random.choice(dates, num_players * num_days),
    "Player_ID": np.random.choice(player_ids, num_players * num_days),
    "Virtual_Item_Purchased": np.random.randint(0, 2, num_players * num_days),  # 0代表未购买，1代表购买
    "Activity_Level": np.random.randint(1, 100, num_players * num_days),  # 活跃度分数
    "Missions_Completed": np.random.randint(0, 10, num_players * num_days)  # 完成的任务数量
}

# 将数据转换为Pandas DataFrame
df_game = pd.DataFrame(data)

# 按日期分组，统计每日的虚拟物品销售数量
sales_trend = df_game.groupby("Date")["Virtual_Item_Purchased"].sum()

# 根据购买虚拟物品与否，将玩家分为两组
buyers = df_game[df_game["Virtual_Item_Purchased"] == 1]
non_buyers = df_game[df_game["Virtual_Item_Purchased"] == 0]

# 计算购买与未购买玩家的平均活跃度和平均任务完成数量
avg_activity_buyers = buyers["Activity_Level"].mean()
avg_missions_buyers = buyers["Missions_Completed"].mean()
avg_activity_non_buyers = non_buyers["Activity_Level"].mean()
avg_missions_non_buyers = non_buyers["Missions_Completed"].mean()

# 使用线性回归模型预测未来的销售情况
df_game["Day"] = (df_game["Date"] - df_game["Date"].min()).dt.days
X = df_game.groupby("Day")[["Virtual_Item_Purchased"]].sum().index.values.reshape(-1, 1)
y = df_game.groupby("Day")[["Virtual_Item_Purchased"]].sum().values
model = LinearRegression()
model.fit(X, y)
future_days = np.array(range(num_days, num_days + 7)).reshape(-1, 1)
predicted_sales = model.predict(future_days)

# 绘制虚拟物品的销售趋势图表
plt.figure(figsize=(12, 6))
sales_trend.plot()
plt.title("虚拟物品的销售趋势", fontproperties=myfont)
plt.xlabel("日期", fontproperties=myfont)
plt.ylabel("销售数量", fontproperties=myfont)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 绘制购买和未购买玩家的活跃度和任务完成数量的对比图
labels = ["购买玩家", "未购买玩家"]
activity_levels = [avg_activity_buyers, avg_activity_non_buyers]
missions_completed = [avg_missions_buyers, avg_missions_non_buyers]
x = np.arange(len(labels))
width = 0.35
plt.figure(figsize=(12, 6))
rects1 = plt.bar(x - width/2, activity_levels, width, label='平均活跃度', alpha=0.8)
rects2 = plt.bar(x + width/2, missions_completed, width, label='平均任务完成数', alpha=0.8)
plt.xlabel('玩家类型', fontproperties=myfont)
plt.ylabel('分数', fontproperties=myfont)
plt.title('购买和未购买玩家的平均活跃度和任务完成数量的对比', fontproperties=myfont)
plt.xticks(x, labels, fontproperties=myfont)
plt.legend(prop=myfont)
plt.tight_layout()
plt.show()

# 绘制未来一周的销售预测图
future_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
plt.figure(figsize=(12, 6))
plt.plot(future_dates, predicted_sales, marker='o')
plt.title("未来一周的销售预测", fontproperties=myfont)
plt.xlabel("日期", fontproperties=myfont)
plt.ylabel("预测销售数量", fontproperties=myfont)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

