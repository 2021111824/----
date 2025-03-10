import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'SimHei'

# 数据：用户数量为 100、150、200，四个算法的响应时间和加权Jain指数
user_counts = [100, 150, 200]

# 假设数据：每个算法在不同用户数量下的平均响应时间 (单位：秒)
greedy_response = [1.2, 1.5, 1.8]
gurobi_response = [1.1, 1.4, 1.7]
ga_response = [1.3, 1.6, 1.9]
no_fairness_response = [2.0, 2.3, 2.6]

# 假设数据：每个算法在不同用户数量下的加权Jain指数 (越接近1越好)
greedy_jain = [0.9696, 0.9756, 0.9795]
gurobi_jain = [0.9973, 0.9889, 0.9955]
ga_jain = [0.7819, 0.7236, 0.6205]
no_fairness_jain = [0.6557, 0.6640, 0.4642]

# 创建响应时间图
fig1, ax1 = plt.subplots(figsize=(10, 6))
bar_width = 2.5  # 增加宽度
ax1.set_xlabel('用户数量')
ax1.set_ylabel('平均响应时间 (秒)', color='tab:blue')

# 设置柱状图
ax1.bar(np.array(user_counts) - bar_width * 1.5, greedy_response, bar_width, label='贪心算法', color='tab:blue', alpha=0.7)
ax1.bar(np.array(user_counts) - bar_width * 0.5, gurobi_response, bar_width, label='Gurobi', color='tab:orange', alpha=0.7)
ax1.bar(np.array(user_counts) + bar_width * 0.5, ga_response, bar_width, label='遗传算法', color='tab:green', alpha=0.7)
ax1.bar(np.array(user_counts) + bar_width * 1.5, no_fairness_response, bar_width, label='无公平性算法', color='tab:red', alpha=0.7)

ax1.tick_params(axis='y', labelcolor='tab:blue')

# 设置横坐标刻度
ax1.set_xticks(user_counts)  # 设置横坐标刻度为100, 150, 200
ax1.set_xticklabels([str(x) for x in user_counts])  # 设置刻度标签

# 设置图例和标题
ax1.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1))
plt.title('不同算法在不同用户数量下的平均响应时间对比')
plt.tight_layout()
plt.show()

# 创建加权Jain指数图
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.set_xlabel('用户数量')
ax2.set_ylabel('加权Jain指数', color='tab:purple')

# 绘制折线图
ax2.plot(user_counts, greedy_jain, label='基于贪心的优化方法', color='tab:blue', marker='o', linestyle='--')
ax2.plot(user_counts, gurobi_jain, label='Gurobi', color='tab:orange', marker='o', linestyle='--')
ax2.plot(user_counts, ga_jain, label='遗传算法', color='tab:green', marker='o', linestyle='--')
ax2.plot(user_counts, no_fairness_jain, label='无公平性算法', color='tab:red', marker='o', linestyle='--')

ax2.tick_params(axis='y', labelcolor='tab:purple')

# 设置横坐标刻度
ax2.set_xticks(user_counts)  # 设置横坐标刻度为100, 150, 200
ax2.set_xticklabels([str(x) for x in user_counts])  # 设置刻度标签

# 设置图例和标题
ax2.legend(loc='center left', bbox_to_anchor=(0.77, 0.7))
plt.title('不同算法在不同用户数量下的加权Jain指数对比')
plt.tight_layout()
plt.show()
