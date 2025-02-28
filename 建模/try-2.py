import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'SimHei'


# 数据：优先级为1, 2, 3
user_counts = [1, 2, 3]
user_pri = ["优先级1", "优先级2", "优先级3"]

# 数据：每个算法在不同优先级的平均响应时间 (单位：秒)
greedy_response_100 = [8.60, 5.51, 4.12]
gurobi_response_100 = [6.73, 3.36, 2.29]
ga_response_100 = [6.64, 5.21, 3.75]
no_fairness_response_100 = [5.02, 4.66, 4.51]

greedy_response_150 = [8.13, 4.65, 3.70]
gurobi_response_150 = [8.11, 4.28, 2.81]
ga_response_150 = [6.57, 5.45, 4.34]
no_fairness_response_150 = [5.15, 4.96, 4.65]

greedy_response_200 = [7.37, 4.22, 3.20]
gurobi_response_200 = [6.62, 3.42, 2.17]
ga_response_200 = [8.89, 6.64, 4.37]
no_fairness_response_200 = [5.81, 6.70, 5.24]

# ========== 用户数量100 ========= #
fig1, ax1 = plt.subplots(figsize=(10, 6))
bar_width = 0.12  # 增加宽度
ax1.set_xlabel('用户优先级')
ax1.set_ylabel('平均响应时间 (ms)', color='tab:blue')

# 设置柱状图
bar1 = ax1.bar(np.array(user_counts) - bar_width * 1.5, greedy_response_100, bar_width, label='贪心算法', color='tab:blue', alpha=0.7)
bar2 = ax1.bar(np.array(user_counts) - bar_width * 0.5, gurobi_response_100, bar_width, label='Gurobi', color='tab:orange', alpha=0.7)
bar3 = ax1.bar(np.array(user_counts) + bar_width * 0.5, ga_response_100, bar_width, label='遗传算法', color='tab:green', alpha=0.7)
bar4 = ax1.bar(np.array(user_counts) + bar_width * 1.5, no_fairness_response_100, bar_width, label='无公平性算法', color='tab:red', alpha=0.7)

# 绘制折线
ax1.plot(user_counts, [bar1[i].get_height() for i in range(len(bar1))], color='tab:blue', marker='o', linestyle='--', markersize=8)
ax1.plot(user_counts, [bar2[i].get_height() for i in range(len(bar2))], color='tab:orange', marker='o', linestyle='--', markersize=8)
ax1.plot(user_counts, [bar3[i].get_height() for i in range(len(bar3))], color='tab:green', marker='o', linestyle='--', markersize=8)
ax1.plot(user_counts, [bar4[i].get_height() for i in range(len(bar4))], color='tab:red', marker='o', linestyle='--', markersize=8)

ax1.tick_params(axis='y', labelcolor='tab:blue')

# 设置横坐标刻度
ax1.set_xticks(user_counts)  # 设置横坐标刻度为1,2,3
ax1.set_xticklabels([str(x) for x in user_pri])  # 设置刻度标签

# 设置图例和标题
ax1.legend(loc='center left', bbox_to_anchor=(0.8, 0.85))
plt.title('用户数量为 100 时各算法不同优先级的平均响应时间对比')
plt.tight_layout()
plt.show()

# ========== 用户数量150 ========= #
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.set_xlabel('用户优先级')
ax2.set_ylabel('平均响应时间 (ms)', color='tab:blue')

# 设置柱状图
bar1 = ax2.bar(np.array(user_counts) - bar_width * 1.5, greedy_response_150, bar_width, label='贪心算法', color='tab:blue', alpha=0.7)
bar2 = ax2.bar(np.array(user_counts) - bar_width * 0.5, gurobi_response_150, bar_width, label='Gurobi', color='tab:orange', alpha=0.7)
bar3 = ax2.bar(np.array(user_counts) + bar_width * 0.5, ga_response_150, bar_width, label='遗传算法', color='tab:green', alpha=0.7)
bar4 = ax2.bar(np.array(user_counts) + bar_width * 1.5, no_fairness_response_150, bar_width, label='无公平性算法', color='tab:red', alpha=0.7)

# 绘制折线
ax2.plot(user_counts, [bar1[i].get_height() for i in range(len(bar1))], color='tab:blue', marker='o', linestyle='--', markersize=8)
ax2.plot(user_counts, [bar2[i].get_height() for i in range(len(bar2))], color='tab:orange', marker='o', linestyle='--', markersize=8)
ax2.plot(user_counts, [bar3[i].get_height() for i in range(len(bar3))], color='tab:green', marker='o', linestyle='--', markersize=8)
ax2.plot(user_counts, [bar4[i].get_height() for i in range(len(bar4))], color='tab:red', marker='o', linestyle='--', markersize=8)

ax2.tick_params(axis='y', labelcolor='tab:blue')

# 设置横坐标刻度
ax2.set_xticks(user_counts)  # 设置横坐标刻度为1,2,3
ax2.set_xticklabels([str(x) for x in user_pri])  # 设置刻度标签

# 设置图例和标题
ax2.legend(loc='center left', bbox_to_anchor=(0.8, 0.85))
plt.title('用户数量为 150 时各算法不同优先级的平均响应时间对比')
plt.tight_layout()
plt.show()

# ========== 用户数量200 ========= #
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.set_xlabel('用户优先级')
ax3.set_ylabel('平均响应时间 (ms)', color='tab:blue')

# 设置柱状图
bar1 = ax3.bar(np.array(user_counts) - bar_width * 1.5, greedy_response_200, bar_width, label='贪心算法', color='tab:blue', alpha=0.7)
bar2 = ax3.bar(np.array(user_counts) - bar_width * 0.5, gurobi_response_200, bar_width, label='Gurobi', color='tab:orange', alpha=0.7)
bar3 = ax3.bar(np.array(user_counts) + bar_width * 0.5, ga_response_200, bar_width, label='遗传算法', color='tab:green', alpha=0.7)
bar4 = ax3.bar(np.array(user_counts) + bar_width * 1.5, no_fairness_response_200, bar_width, label='无公平性算法', color='tab:red', alpha=0.7)

# 绘制折线
ax3.plot(user_counts, [bar1[i].get_height() for i in range(len(bar1))], color='tab:blue', marker='o', linestyle='--', markersize=8)
ax3.plot(user_counts, [bar2[i].get_height() for i in range(len(bar2))], color='tab:orange', marker='o', linestyle='--', markersize=8)
ax3.plot(user_counts, [bar3[i].get_height() for i in range(len(bar3))], color='tab:green', marker='o', linestyle='--', markersize=8)
ax3.plot(user_counts, [bar4[i].get_height() for i in range(len(bar4))], color='tab:red', marker='o', linestyle='--', markersize=8)

ax3.tick_params(axis='y', labelcolor='tab:blue')

# 设置横坐标刻度
ax3.set_xticks(user_counts)  # 设置横坐标刻度为1,2,3
ax3.set_xticklabels([str(x) for x in user_pri])  # 设置刻度标签

# 设置图例和标题
ax3.legend(loc='center left', bbox_to_anchor=(0.8, 0.85))
plt.title('用户数量为 200 时各算法不同优先级的平均响应时间对比')
plt.tight_layout()
plt.show()
