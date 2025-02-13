# 可视化模块--visualization.py
# 实现绘图相关的功能
import os
import matplotlib.pyplot as plt
import numpy as np


# ========== 可视化模块 ==========

def add_bar_labels(bars, values, precision=2):
    """
    在柱形图顶部标注数据
    Args:
        bars: plt.bar() 的返回值
        values: 与 bars 对应的值列表
        precision: 显示小数点后的精度
    """
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{value:.{precision}f}", ha='center', va='bottom', fontsize=10)


def save_priority_distribution(priorities, output_folder):
    """
    保存优先级分布到文本文件
    """
    unique, counts = np.unique(priorities, return_counts=True)
    priority_distribution = dict(zip(unique, counts))

    # 保存到文件
    with open(os.path.join(output_folder, "priority_distribution.txt"), "w") as f:
        f.write("===== Priority Distribution =====\n")
        for priority, count in priority_distribution.items():
            f.write(f"Priority {priority}: {count} users\n")
    print("Priority distribution saved.")


def plot_user_server_distribution(user_positions, server_positions, priorities, m_edge, output_folder):
    """
    绘制用户和服务器的分布
    """
    colors = ['blue', 'orange', 'purple']
    plt.figure(figsize=(10, 6))

    for level in range(1, 4):
        idx = np.where(priorities == level)
        plt.scatter(user_positions[idx, 0], user_positions[idx, 1], c=colors[level - 1], label=f"Priority {level}", alpha=0.7, s=50)

    plt.scatter(server_positions[:m_edge, 0], server_positions[:m_edge, 1], c='green', label='Edge Servers', marker='^', s=100)
    plt.scatter(server_positions[m_edge:, 0], server_positions[m_edge:, 1], c='red', label='Cloud Servers', marker='s', s=150)

    plt.title("User and Server Distribution by Priorities")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(output_folder, "user_server_distribution_priorities.png"))
    plt.show()


def plot_fitness_history(fitness_history, output_folder):
    """
    绘制适应度随代数变化曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fitness_history)), fitness_history, label="fitness")
    plt.title("Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(output_folder, "fitness_history.png"))
    plt.show()


def plot_jain_history(best_jain_fairness_history, output_folder):
    """
    绘制Jain公平性指数随代数变化曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(best_jain_fairness_history)), best_jain_fairness_history, label="Jain Fairness Index")
    plt.title("Best Jain Fairness Index Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Jain Fairness Index")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(output_folder, "jain_fairness_history.png"))
    plt.show()


def plot_response_deviation_history(best_response_deviation_history, output_folder):
    """
    绘制响应时间偏差随代数变化曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(best_response_deviation_history)), best_response_deviation_history, label="response_deviation")
    plt.title("response_deviation Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("response_deviation")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(output_folder, "response_deviation.png"))
    plt.show()


def plot_response_time_distribution(response_times, priorities, output_folder):
    """
    绘制响应时间分布，包括整体分布和按优先级的分布
    """
    response_times = np.array(response_times)  # 将响应时间转换为 NumPy 数组
    priorities = np.array(priorities)  # 将优先级也转换为 NumPy 数组
    colors = ['blue', 'orange', 'purple']  # 为每个优先级分配颜色

    # 确定响应时间的最大最小值，以确保每个优先级使用相同的分箱
    min_time = np.min(response_times)
    max_time = np.max(response_times)
    bins = np.linspace(min_time, max_time, 20)  # 使用固定的 20 个分箱

    # 绘制按优先级分布
    plt.figure(figsize=(12, 6))
    for level in np.unique(priorities):  # 遍历每种优先级
        idx = priorities == level  # 找到当前优先级的布尔索引
        times = response_times[idx]  # 使用布尔索引提取响应时间
        plt.hist(times, bins=bins, alpha=0.6, label=f"Priority {level}", color=colors[level - 1])

    plt.title("Response Time Distribution by Priority")
    plt.xlabel("Response Time (ms)")
    plt.ylabel("Number of Users")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(output_folder, "response_time_distribution_by_priority.png"))
    plt.show()

    # 绘制整体响应时间分布
    plt.figure(figsize=(10, 6))
    plt.hist(response_times, bins=bins, color='gray', edgecolor='black', alpha=0.7)
    plt.title("Overall Response Time Distribution")
    plt.xlabel("Response Time (ms)")
    plt.ylabel("Number of Users")
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(output_folder, "response_time_distribution_overall.png"))
    plt.show()


def plot_avg_response_time(response_times, priorities, output_folder, T_max):
    """
    绘制平均响应时间柱状图，并标注优先级用户的响应时间上限，用对应颜色区分。

    Args:
        response_times (list): 用户响应时间
        priorities (list): 用户优先级
        output_folder (str): 输出文件夹路径
        T_max (dict): 不同优先级的响应时间上限
    """
    response_times = np.array(response_times)
    priorities = np.array(priorities)
    avg_response_per_priority = {}

    # 定义颜色，与柱状图一致
    colors = {1: 'blue', 2: 'orange', 3: 'purple'}

    # 计算每个优先级的平均响应时间
    for level in np.unique(priorities):
        idx = np.where(priorities == level)
        avg_response_per_priority[level] = np.mean(response_times[idx])

    levels = list(avg_response_per_priority.keys())
    avg_responses = list(avg_response_per_priority.values())

    # 创建柱形图
    plt.figure(figsize=(8, 5))
    bars = plt.bar(levels, avg_responses, color=[colors[level] for level in levels], alpha=0.7)

    # 添加柱形顶部标注
    for bar, value in zip(bars, avg_responses):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{value:.2f}", ha='center', va='bottom', fontsize=10)

    # 添加响应时间上限的水平线，用不同颜色
    for level, max_time in T_max.items():
        plt.axhline(y=max_time, color=colors[level], linestyle='--', linewidth=1.5,
                    label=f"Priority {level} Max ({max_time} ms)")

    # 图表标题和标签
    plt.title("Average Response Time by Priority")
    plt.xlabel("Priority Level")
    plt.ylabel("Average Response Time (ms)")
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(output_folder, "avg_response_time.png"))
    plt.show()


def plot_server_resource_usage(server_cpu_usage, server_mem_usage, server_bandwidth_usage,
                               R_cpu, R_mem, R_bandwidth, m_edge, output_folder):
    """
    绘制服务器资源利用率，包括 CPU、内存和带宽，并区分边缘和云服务器。
    同时显示每个资源使用占该服务器最大资源的百分比，并显示整体资源利用率。
    """
    n_servers = len(server_cpu_usage)
    server_indices = np.arange(n_servers)

    # 边缘和云服务器资源上限
    max_edge_cpu = max(R_cpu[:m_edge])
    max_cloud_cpu = max(R_cpu[m_edge:])
    max_edge_mem = max(R_mem[:m_edge])
    max_cloud_mem = max(R_mem[m_edge:])
    max_edge_bandwidth = max(R_bandwidth[:m_edge])
    max_cloud_bandwidth = max(R_bandwidth[m_edge:])

    # 计算资源使用占比（百分比）
    cpu_percentage = np.array(server_cpu_usage) / np.array(R_cpu) * 100
    mem_percentage = np.array(server_mem_usage) / np.array(R_mem) * 100
    bandwidth_percentage = np.array(server_bandwidth_usage) / np.array(R_bandwidth) * 100

    # 计算边缘服务器的整体资源利用率
    total_edge_cpu_usage = np.sum(server_cpu_usage[:m_edge])
    total_edge_mem_usage = np.sum(server_mem_usage[:m_edge])
    total_edge_bandwidth_usage = np.sum(server_bandwidth_usage[:m_edge])

    total_edge_cpu_max = np.sum(R_cpu[:m_edge])
    total_edge_mem_max = np.sum(R_mem[:m_edge])
    total_edge_bandwidth_max = np.sum(R_bandwidth[:m_edge])

    edge_cpu_usage_rate = total_edge_cpu_usage / total_edge_cpu_max * 100
    edge_mem_usage_rate = total_edge_mem_usage / total_edge_mem_max * 100
    edge_bandwidth_usage_rate = total_edge_bandwidth_usage / total_edge_bandwidth_max * 100

    # 绘制 CPU 使用率
    plt.figure(figsize=(10, 6))
    plt.bar(server_indices[:m_edge], server_cpu_usage[:m_edge], color='blue', alpha=0.7, label="Edge Servers")
    plt.bar(server_indices[m_edge:], server_cpu_usage[m_edge:], color='red', alpha=0.7, label="Cloud Servers")
    plt.axhline(y=max_edge_cpu, color='blue', linestyle='--', label="Max Edge CPU")
    plt.axhline(y=max_cloud_cpu, color='red', linestyle='--', label="Max Cloud CPU")
    plt.text(m_edge - 1, max_edge_cpu + 1, f"Edge Total CPU Usage: {edge_cpu_usage_rate:.1f}%", ha='center',
             color='black', fontsize=10)

    # 添加百分比标签
    for i in range(m_edge):
        plt.text(i, server_cpu_usage[i] + 0.5, f"{cpu_percentage[i]:.1f}%", ha='center', color='black', fontsize=7)
    for i in range(m_edge, n_servers):
        plt.text(i, server_cpu_usage[i] + 0.5, f"{cpu_percentage[i]:.1f}%", ha='center', color='black', fontsize=7)

    plt.title("Server CPU Usage")
    plt.xlabel("Server Index")
    plt.ylabel("CPU Usage")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(output_folder, "server_cpu_usage.png"))
    plt.show()

    # 绘制内存使用率
    plt.figure(figsize=(10, 6))
    plt.bar(server_indices[:m_edge], server_mem_usage[:m_edge], color='blue', alpha=0.7, label="Edge Servers")
    plt.bar(server_indices[m_edge:], server_mem_usage[m_edge:], color='red', alpha=0.7, label="Cloud Servers")
    plt.axhline(y=max_edge_mem, color='blue', linestyle='--', label="Max Edge Memory")
    plt.axhline(y=max_cloud_mem, color='red', linestyle='--', label="Max Cloud Memory")
    plt.text(m_edge - 1, max_edge_mem + 1, f"Edge Total Memory Usage: {edge_mem_usage_rate:.1f}%", ha='center',
             color='black', fontsize=10)

    # 添加百分比标签
    for i in range(m_edge):
        plt.text(i, server_mem_usage[i] + 0.5, f"{mem_percentage[i]:.1f}%", ha='center', color='black', fontsize=7)
    for i in range(m_edge, n_servers):
        plt.text(i, server_mem_usage[i] + 0.5, f"{mem_percentage[i]:.1f}%", ha='center', color='black', fontsize=7)

    plt.title("Server Memory Usage")
    plt.xlabel("Server Index")
    plt.ylabel("Memory Usage (GB)")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(output_folder, "server_mem_usage.png"))
    plt.show()

    # 绘制带宽使用率
    plt.figure(figsize=(10, 6))
    plt.bar(server_indices[:m_edge], server_bandwidth_usage[:m_edge], color='blue', alpha=0.7, label="Edge Servers")
    plt.bar(server_indices[m_edge:], server_bandwidth_usage[m_edge:], color='red', alpha=0.7, label="Cloud Servers")
    plt.axhline(y=max_edge_bandwidth, color='blue', linestyle='--', label="Max Edge Bandwidth")
    plt.axhline(y=max_cloud_bandwidth, color='red', linestyle='--', label="Max Cloud Bandwidth")
    plt.text(m_edge - 1, max_edge_bandwidth + 1, f"Edge Total Bandwidth Usage: {edge_bandwidth_usage_rate:.1f}%",
             ha='center', color='black', fontsize=10)

    # 添加百分比标签
    for i in range(m_edge):
        plt.text(i, server_bandwidth_usage[i] + 0.5, f"{bandwidth_percentage[i]:.1f}%", ha='center', color='black', fontsize=7)
    for i in range(m_edge, n_servers):
        plt.text(i, server_bandwidth_usage[i] + 0.5, f"{bandwidth_percentage[i]:.1f}%", ha='center', color='black', fontsize=7)

    plt.title("Server Bandwidth Usage")
    plt.xlabel("Server Index")
    plt.ylabel("Bandwidth Usage (Mbps)")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(output_folder, "server_bandwidth_usage.png"))
    plt.show()


def plot_cost_distribution(cost_details, output_folder, total_edge_cost, total_cloud_cost, total_cost, cost_limit):
    """
    绘制边缘服务器和云服务器的成本分布，并在图表下方标注汇总信息。
    """
    # 定义固定顺序
    categories = ["Cpu", "Mem", "Bandwidth", "Fixed", "Network"]
    edge_costs = [cost_details["edge"].get(cat.lower(), 0) for cat in categories]  # 边缘节点按顺序获取成本
    cloud_costs = [cost_details["cloud"].get(cat.lower(), 0) for cat in categories]  # 云节点按顺序获取成本

    # 生成柱形图
    x = np.arange(len(categories))  # 统一使用总类别数
    width = 0.4  # 调整柱宽使其更美观

    plt.figure(figsize=(14, 6))
    bars_edge = plt.bar(x, edge_costs, width, label="Edge Servers", color='blue')  # 边缘节点柱形图
    bars_cloud = plt.bar(x, cloud_costs, width, label="Cloud Servers", color='red', alpha=0.7)  # 云节点柱形图，透明度略低

    # 添加柱形顶部标注
    add_bar_labels(bars_edge, edge_costs)
    add_bar_labels(bars_cloud, cloud_costs)

    # 添加汇总信息到图表下方
    summary_text = (
        f"Edge Total Cost: {total_edge_cost:.2f}  |  "
        f"Cloud Total Cost: {total_cloud_cost:.2f}  |  "
        f"Overall Total Cost: {total_cost:.2f}  |  "
        f"Cost Limit: {cost_limit:.2f}"
    )
    plt.text(0.5, -0.15, summary_text, ha='center', va='top', transform=plt.gca().transAxes, fontsize=12)

    # 设置X轴标签
    plt.xlabel("Cost Categories")
    plt.ylabel("Cost (Units)")
    plt.title("Cost Breakdown by Server Type")
    plt.xticks(x, categories)  # 按固定顺序设置类别名称
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()  # 自动调整布局以适应新增内容
    plt.savefig(os.path.join(output_folder, "cost_distribution.png"))
    plt.show()


def plot_user_server_connections(user_positions, server_positions, best_solution, priorities, m_edge, output_folder):
    """
    绘制用户与服务器的连接
    """
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'orange', 'purple']

    for level in range(1, 4):
        idx = np.where(priorities == level)
        plt.scatter(user_positions[idx, 0], user_positions[idx, 1], c=colors[level - 1], label=f"Priority {level}", alpha=0.7, s=50)

    plt.scatter(server_positions[:m_edge, 0], server_positions[:m_edge, 1], c='green', label='Edge Servers', marker='^', s=100)
    plt.scatter(server_positions[m_edge:, 0], server_positions[m_edge:, 1], c='red', label='Cloud Servers', marker='s', s=150)

    for i, user in enumerate(user_positions):
        server_idx = np.argmax(best_solution[i])
        plt.plot([user[0], server_positions[server_idx, 0]], [user[1], server_positions[server_idx, 1]], color=colors[priorities[i] - 1], alpha=0.3)

    plt.title("User-to-Server Connections by Priorities")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()  # 自动调整图形布局
    plt.savefig(os.path.join(output_folder, "user_server_connections.png"))
    plt.show()
