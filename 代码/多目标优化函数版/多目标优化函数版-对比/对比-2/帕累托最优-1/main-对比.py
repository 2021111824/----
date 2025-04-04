# 主程序模块--main.py
import os
import numpy as np
import json

from calculations import calculate_response_stats, calculate_total_cost, assign_computational_capacity, \
    compute_response_time
from genetic_algorithm import nsga_ii, compute_objectives, compute_jain_fairness
from visualization import save_priority_distribution, plot_user_server_distribution, plot_fitness_history, \
    plot_response_time_distribution, plot_avg_response_time, plot_server_resource_usage, plot_user_server_connections, \
    plot_cost_distribution


def load_simulation_data():
    with open('simulation_data.json', 'r') as f:
        return json.load(f)


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
    # 加载模拟数据
    simulation_data = load_simulation_data()

    # 提取封装好的数据
    user_positions = simulation_data['user_positions']
    request_sizes = simulation_data['request_sizes']
    priorities = simulation_data['priorities']
    server_positions = simulation_data['server_positions']
    R_cpu = simulation_data['R_cpu']
    R_mem = simulation_data['R_mem']
    R_bandwidth = simulation_data['R_bandwidth']
    cpu_demands = simulation_data['cpu_demands']
    mem_demands = simulation_data['mem_demands']
    bandwidth_demands = simulation_data['bandwidth_demands']

    # 转换为 NumPy 数组
    server_positions = np.array(server_positions)
    user_positions = np.array(user_positions)
    priorities = np.array(priorities)

    edge_positions = server_positions[:m_edge]
    cloud_positions = server_positions[m_edge:]

    # 验证优先级分布
    save_priority_distribution(priorities, output_folder)

    # 绘制用户和服务器的初始分布
    plot_user_server_distribution(user_positions, server_positions, priorities, m_edge, output_folder)

    # ==========  NSGA - II 算法求解 ==========
    pareto_front = nsga_ii(
        user_positions, server_positions, request_sizes, priorities,
        R_cpu, R_mem, R_bandwidth,
        cost_edge, cost_cloud,
        m_edge, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud,
        P, G_max, P_c, P_m, max_cost,
        cpu_demands, mem_demands, bandwidth_demands, p_net,
        T_max
    )

    # 使用加权和法选择一个最终解
    w1 = 0.5  # Jain 公平性指数的权重
    w2 = 0.5  # 响应时间偏差比的权重
    best_score = -np.inf  # 负无穷
    best_solution = None
    best_jain_fairness = None
    best_response_deviation = None
    best_response_times = None

    for solution in pareto_front:
        jain, deviation = compute_objectives(solution, user_positions, server_positions, request_sizes, priorities,
                                             v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)
        score = w1 * (-jain) + w2 * (-deviation)  # 注意这里 -jain 是因为之前计算时取了负号
        if score > best_score:
            best_score = score
            best_solution = solution
            best_jain_fairness = -jain  # 恢复 Jain 公平性指数的原始值
            best_response_deviation = deviation

    best_response_times = []
    n_users = len(user_positions)
    user_capacities = assign_computational_capacity(best_solution, user_positions, server_positions, request_sizes,
                                                    P_edge, P_cloud, m_edge, priorities)
    for i in range(n_users):
        server_idx = np.argmax(best_solution[i])  # 用户分配到的服务器
        is_edge = server_idx < m_edge  # 是否为边缘服务器
        response_time = compute_response_time(
            user_positions[i], server_positions[server_idx], is_edge, request_sizes[i], user_capacities[i],
            v_edge, v_cloud, b_edge, b_cloud
        )
        best_response_times.append(response_time)

    # ========== 结果分析 ==========

    # 平均响应时间
    avg_response_time = np.mean(best_response_times)

    # 响应时间统计
    response_stats = calculate_response_stats(best_response_times, priorities)

    # 计算每个优先级的 Jain 公平性指数
    jain_fairness_per_priority = {}
    unique_priorities = np.unique(priorities)
    for priority in unique_priorities:
        priority_indices = [i for i, p in enumerate(priorities) if p == priority]
        priority_response_times = [best_response_times[i] for i in priority_indices]
        jain_fairness_per_priority[priority] = compute_jain_fairness(priority_response_times)

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
        f.write(f"Best Score: {best_score:.4f}\n")

        f.write("Jain Fairness per Priority:\n")
        for priority, fairness in jain_fairness_per_priority.items():
            f.write(f"  Priority {priority}: {fairness:.4f}\n")

        f.write(f"Overall Jain Fairness: {best_jain_fairness:.4f}\n")

        f.write(f"Response Deviation : {best_response_deviation:.4f}\n")

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
    # plot_fitness_history(fitness_history, output_folder)
    # Jain公平性指数随代数变化曲线
    # plot_jain_history(best_jain_fairness_history, output_folder)
    # # 响应时间偏差随代数变化曲线
    # plot_response_deviation_history(best_response_deviation_history, output_folder)

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
