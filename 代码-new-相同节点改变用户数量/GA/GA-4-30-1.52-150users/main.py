# 主程序模块--main.py
import os
import numpy as np
import time

from calculations import calculate_response_stats, calculate_total_cost
from genetic_algorithm import genetic_algorithm
from initialize import initialize_topology
from visualization import save_priority_distribution, plot_user_server_distribution, plot_fitness_history, \
    plot_response_time_distribution, plot_avg_response_time, plot_server_resource_usage, plot_user_server_connections, \
    plot_cost_distribution, plot_service_instance_distribution

# ========== 主程序入口 ==========
if __name__ == "__main__":

    # 记录程序开始时间
    start_time = time.time()

    # ========== 数据初始化 ==========
    n, user_positions, priorities, weights, user_data, p_user, P_allocation, T_max, \
        m_edge, m_cloud, server_positions, t_delay_e, t_delay_c, R_bandwidth, R_edge, P_edge, P_cloud, \
        p_m, r_m, cost_edge, cost_cloud, max_cost, Population, G_max, P_crossover, P_mutation = initialize_topology()

    edge_positions = server_positions[:m_edge]
    cloud_positions = server_positions[m_edge:]

    # 输出文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # 验证优先级分布
    save_priority_distribution(priorities, output_folder)

    # 绘制用户和服务器的初始分布
    plot_user_server_distribution(user_positions, server_positions, priorities, m_edge, output_folder)

    # ========== 运行遗传算法 ==========
    best_solution, best_fitness, best_response_times, fitness_history = genetic_algorithm(
        n, m_edge, m_cloud, priorities, weights, R_bandwidth, cost_edge, cost_cloud,
        Population, G_max, P_crossover, P_mutation, max_cost, T_max, p_user, p_m, r_m, R_edge,
        t_delay_e, t_delay_c, user_data, P_allocation)

    # ========== 结果分析 ==========

    # 平均响应时间
    avg_response_time = np.mean(best_response_times)

    # 响应时间统计
    response_stats = calculate_response_stats(best_response_times, priorities)

    # 记录资源使用情况
    server_compute_capability = np.zeros(len(server_positions))
    server_compute_resource_usage = np.zeros(len(server_positions))

    # 初始化每个服务器上部署的服务实例数量
    service_instances = np.zeros(len(server_positions))  # 存储每个服务器上的服务实例数量

    # 遍历用户，统计每台服务器的资源使用
    for i in range(len(user_positions)):
        server_idx = np.argmax(best_solution[i])  # 获取用户分配到的服务器
        is_edge = server_idx < m_edge  # 判断是否是边缘服务器

        # 使用用户的计算能力需求更新服务器计算能力需求
        server_compute_capability[server_idx] += p_user[i]  # 服务器上的用户计算能力需求

    # 服务器上的计算资源使用情况
    server_compute_resource_usage = (np.ceil(server_compute_capability / p_m)) * r_m  # 服务器的计算资源使用情况

    # 计算每个服务器上部署的服务实例数量
    service_instances = server_compute_resource_usage / r_m

    # 计算总成本和分项成本
    total_cost, cost_details = calculate_total_cost(best_solution, m_edge, cost_edge, cost_cloud, p_user)

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

    # 计算并保存每个服务器的服务实例数量到文件
    with open(os.path.join(output_folder, "service_instances.txt"), "w") as f:
        f.write("===== Service Instances per Server =====\n")
        for j in range(len(server_positions)):
            server_type = "Edge" if j < m_edge else "Cloud"
            f.write(f"Server {j} ({server_type}): {service_instances[j]} instances\n")

    print(f"Service instance results saved to '{output_folder}/service_instances.txt'.")

    # 保存详细结果到文件
    with open(os.path.join(output_folder, "simulation_results.txt"), "w") as f:
        f.write("===== Simulation Results =====\n")
        f.write(f"Best Fitness (Jain Fairness Index): {best_fitness:.4f}\n")

        f.write(f"Average Response Time of Best Solution: {avg_response_time:.2f} ms\n\n")

        f.write(f"Response Time Statistics by Priority:\n")
        for level, stats in response_stats.items():
            status = "OK" if stats["mean"] <= T_max[level] else "EXCEEDED"
            f.write(f"  Priority {level}: Mean={stats['mean']:.2f} ms (Limit: {T_max[level]} ms) [{status}]\n")

        f.write("\nUser-to-Server Assignment:\n")
        for i in range(len(user_positions)):
            server_idx = np.argmax(best_solution[i])
            server_type = "Edge" if server_idx < m_edge else "Cloud"
            f.write(f"  User {i} -> Server {server_idx} ({server_type})\n")

        # ========== 可视化 ==========
        # 1. 绘制适应度变化曲线
        plot_fitness_history(fitness_history, output_folder)

        # 2. 绘制响应时间分布
        plot_response_time_distribution(best_response_times, priorities, output_folder)

        # 3. 绘制平均响应时间柱状图
        plot_avg_response_time(best_response_times, priorities, output_folder, T_max)

        # 4. 绘制服务器资源使用情况
        plot_server_resource_usage(server_compute_resource_usage, R_edge, m_edge, output_folder)

        # 5. 绘制用户和服务器的连接图
        plot_user_server_connections(user_positions, server_positions, best_solution, priorities, m_edge, output_folder)

        # 6. 绘制服务器部署成本图
        plot_cost_distribution(cost_details, output_folder,
                               total_edge_cost=cost_details['edge']['fixed'],
                               total_cloud_cost=cost_details['cloud']['p_net'],
                               total_cost=total_cost,
                               cost_limit=max_cost)

        # 7. 绘制服务器上的服务实例部署情况
        plot_service_instance_distribution(service_instances, output_folder)

    # 记录程序结束时间
    end_time = time.time()

    # 输出程序总运行时间
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")
