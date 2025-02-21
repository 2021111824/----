# 主程序模块--main.py
import os
import numpy as np

from calculations import calculate_response_stats, calculate_total_cost
from genetic_algorithm import genetic_algorithm
from initialize import initialize_topology
from visualization import save_priority_distribution, plot_user_server_distribution, plot_fitness_history, \
    plot_response_time_distribution, plot_avg_response_time, plot_server_resource_usage, plot_user_server_connections, \
    plot_cost_distribution, plot_jain_history, plot_response_deviation_history

# random.seed(30)
# np.random.seed(30)

# ========== 主程序入口 ==========
if __name__ == "__main__":
    # ========== 参数设置 ==========
    n, m_edge, m_cloud = 150, 20, 3  # 用户数、边缘服务器数、云服务器数
    v_edge, v_cloud = 10.0, 5.0  # 边缘服务器和云服务器的网络传播速度 (Mbps)
    b_edge, b_cloud = 100.0, 500.0  # 边缘和云服务器的带宽速度  (MB/s)
    P_edge, P_cloud = 600.0, 1200.0  # 边缘和云服务器的计算能力 (MB/s)
    P = 100  # 初始种群大小
    G_max = 200  # 最大迭代代数
    P_c, P_m = 0.8, 0.1  # 交叉概率和变异概率
    T_max = {
        1: 16,  # 优先级 1 用户最大允许响应时间 (ms)
        2: 12,  # 优先级 2 用户最大允许响应时间 (ms)
        3: 8,  # 优先级 3 用户最大允许响应时间 (ms)
    }

    # 成本参数
    monthly_fixed_cost = 20.0  # 每月固定成本（单位：某种货币）
    daily_fixed_cost = monthly_fixed_cost / 30.0  # 每日固定成本
    cost_edge = {"fixed": daily_fixed_cost, "cpu": 0.5, "mem": 0.3, "bandwidth": 0.1}  # 边缘服务器成本
    cost_cloud = {"cpu": 0.8, "mem": 0.5, "bandwidth": 0.2}  # 云服务器成本
    p_net = 0.5  # 网络流量单位成本
    max_cost = 1000  # 最大允许总成本

    # 输出文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # ========== 数据初始化 ==========
    user_positions, request_sizes, priorities, server_positions, R_cpu, R_mem, R_bandwidth, cpu_demands, \
        mem_demands, bandwidth_demands = initialize_topology(n, m_edge, m_cloud)
    edge_positions = server_positions[:m_edge]
    cloud_positions = server_positions[m_edge:]

    # 验证优先级分布
    save_priority_distribution(priorities, output_folder)

    # 绘制用户和服务器的初始分布
    plot_user_server_distribution(user_positions, server_positions, priorities, m_edge, output_folder)

    # ========== 遗传算法求解 ==========
    best_solution, best_fitness, jain_fairness, response_deviation, best_response_times, fitness_history, best_jain_fairness_history, \
        best_response_deviation_history = genetic_algorithm(
        user_positions, server_positions, request_sizes, priorities,
        R_cpu, R_mem, R_bandwidth,
        cost_edge, cost_cloud,
        m_edge, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud,
        P, G_max, P_c, P_m, max_cost,
        cpu_demands, mem_demands, bandwidth_demands, p_net,
        T_max
    )

    # ========== 结果分析 ==========

    # 平均响应时间
    avg_response_time = np.mean(best_response_times)

    # 响应时间统计
    response_stats = calculate_response_stats(best_response_times, priorities)

    # 记录资源使用情况
    server_cpu_usage = np.zeros(len(server_positions))
    server_mem_usage = np.zeros(len(server_positions))
    server_bandwidth_usage = np.zeros(len(server_positions))
    request_sizes_per_server = np.zeros(len(server_positions))

    # 遍历用户，统计每台服务器的资源使用
    for i in range(len(user_positions)):
        server_idx = np.argmax(best_solution[i])  # 获取用户分配到的服务器
        is_edge = server_idx < m_edge  # 判断是否是边缘服务器

        # 使用用户的实际资源需求更新服务器资源使用
        server_cpu_usage[server_idx] += cpu_demands[i]  # 用户的 CPU 需求
        server_mem_usage[server_idx] += mem_demands[i]  # 用户的内存需求
        server_bandwidth_usage[server_idx] += bandwidth_demands[i]  # 用户的带宽需求

    # 计算总成本和分项成本
    total_cost, cost_details = calculate_total_cost(
        best_solution,  # 用户到服务器的分配矩阵
        m_edge,  # 边缘服务器数量
        cost_edge,  # 边缘服务器的成本
        cost_cloud,  # 云服务器的成本
        cpu_demands,  # 每个用户的CPU需求
        mem_demands,  # 每个用户的内存需求
        bandwidth_demands,  # 每个用户的带宽需求
        request_sizes,  # 用户的请求大小
        p_net  # 网络传输单位成本
    )

    # 保存到文件
    with open(os.path.join(output_folder, "cost_results.txt"), "w") as f:
        f.write("===== Cost Results =====\n")
        f.write(f"Total Cost: {total_cost:.2f}\n\n")
        f.write(f"Cost Breakdown:\n")
        f.write("  Edge Servers:\n")
        for key, value in cost_details["edge"].items():
            f.write(f"    {key.capitalize()}: {value:.2f}\n")
        f.write("  Cloud Servers:\n")
        for key, value in cost_details["cloud"].items():
            f.write(f"    {key.capitalize()}: {value:.2f}\n")

    # 保存详细结果到文件
    with open(os.path.join(output_folder, "simulation_results.txt"), "w") as f:
        f.write("===== Simulation Results =====\n")
        f.write(f"Best Fitness : {best_fitness:.4f}\n")

        f.write("Jain Fairness per Priority:\n")
        for priority, fairness in jain_fairness.items():
            f.write(f"  Priority {priority}: {fairness:.4f}\n")  # 格式化每个值

        f.write(f"Response Deviation : {response_deviation:.4f}\n")

        f.write(f"Average Response Time of Best Solution: {avg_response_time:.2f} ms\n")

        f.write(f"Response Time Statistics by Priority:\n")
        for level, stats in response_stats.items():
            status = "OK" if stats["mean"] <= T_max[level] else "EXCEEDED"
            f.write(f"  Priority {level}: Mean={stats['mean']:.2f} ms (Limit: {T_max[level]} ms) [{status}]\n")

        f.write("\nUser-to-Server Assignment:\n")
        for i in range(len(user_positions)):
            server_idx = np.argmax(best_solution[i])
            server_type = "Edge" if server_idx < m_edge else "Cloud"
            f.write(f"  User {i} -> Server {server_idx} ({server_type})\n")

    print(f"Simulation results saved to '{output_folder}'.")

    # ========== 可视化 ==========
    # 1.绘制各种变化曲线
    # 适应度变化曲线
    plot_fitness_history(fitness_history, output_folder)
    # Jain公平性指数随代数变化曲线
    plot_jain_history(best_jain_fairness_history, output_folder)
    # 响应时间偏差随代数变化曲线
    plot_response_deviation_history(best_response_deviation_history, output_folder)

    # 2. 绘制响应时间分布
    plot_response_time_distribution(best_response_times, priorities, output_folder)

    # 3. 绘制平均响应时间柱状图
    plot_avg_response_time(best_response_times, priorities, output_folder, T_max)

    # 4. 绘制服务器资源使用情况
    plot_server_resource_usage(server_cpu_usage, server_mem_usage, server_bandwidth_usage,
                               R_cpu, R_mem, R_bandwidth, m_edge, output_folder)

    # 5. 绘制用户和服务器的连接图
    plot_user_server_connections(user_positions, server_positions, best_solution, priorities, m_edge, output_folder)

    # 6. 绘制服务器部署成本图
    plot_cost_distribution(cost_details, output_folder,
                           total_edge_cost=cost_details['edge']['fixed'] + cost_details['edge']['cpu'] +
                                           cost_details['edge']['mem'] + cost_details['edge']['bandwidth'],
                           total_cloud_cost=cost_details['cloud']['cpu'] + cost_details['cloud']['mem'] +
                                            cost_details['cloud']['bandwidth'] + cost_details['cloud']['network'],
                           total_cost=total_cost,
                           cost_limit=max_cost)
