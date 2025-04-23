# 主程序模块--main.py
import os
import numpy as np
import time

from calculations import calculate_response_stats, calculate_total_cost, assign_bandwidth_capacity, \
    compute_response_time
from greedy_algorithm import greedy_algorithm, calculate_weighted_jain_index
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
        m_edge, m_cloud, server_positions, t_delay_e, t_delay_c, R_bandwidth, R_edge, P_edge, \
        p_m, r_m, cost_edge, cost_cloud, max_cost = initialize_topology()

    edge_positions = server_positions[:m_edge]
    cloud_positions = server_positions[m_edge:]

    # 输出文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # 验证优先级分布
    save_priority_distribution(priorities, output_folder)

    # 绘制用户和服务器的初始分布
    plot_user_server_distribution(user_positions, server_positions, priorities, m_edge, output_folder)

    # ========== 运行贪心算法 ==========
    result = greedy_algorithm(n, m_edge, m_cloud, priorities, weights, cost_edge, cost_cloud, max_cost, T_max,
                              R_bandwidth, t_delay_e, t_delay_c, p_m, r_m, R_edge, user_data, p_user, P_allocation)

    # ========== 结果分析 ==========
    if result is None:
        print("未找到满足约束条件的分配方案。")
    else:
        response_times = []
        user_bandwidth = assign_bandwidth_capacity(result, n, m_edge, m_cloud, user_data, R_bandwidth)
        for i in range(len(user_positions)):
            server_idx = np.argmax(result[i])
            is_edge = server_idx < m_edge
            response_time = compute_response_time(t_delay_e[i][server_idx], t_delay_c[i], is_edge, user_data[i], user_bandwidth[i], p_user[i], P_allocation[i])
            response_times.append(response_time)
        print(response_times)

        # 平均响应时间
        avg_response_time = np.mean(response_times)

        # 响应时间统计
        response_stats = calculate_response_stats(response_times, priorities)

        # 计算加权Jain公平性指数
        F_jain = calculate_weighted_jain_index(result, n, m_edge, m_cloud, weights, t_delay_e, t_delay_c, user_data, R_bandwidth,
                                               p_user, P_allocation)

        # 记录资源使用情况
        server_compute_capability = np.zeros(len(server_positions))
        server_compute_resource_usage = np.zeros(len(server_positions))

        # 初始化每个服务器上部署的服务实例数量
        service_instances = np.zeros(len(server_positions))  # 存储每个服务器上的服务实例数量

        # 遍历用户，统计每台服务器的资源使用
        for i in range(len(user_positions)):
            server_idx = np.argmax(result[i])  # 获取用户分配到的服务器
            is_edge = server_idx < m_edge  # 判断是否是边缘服务器

            # 使用用户的计算能力需求更新服务器计算能力需求
            server_compute_capability[server_idx] += p_user[i]  # 服务器上的用户计算能力需求

        # 服务器上的计算资源使用情况
        server_compute_resource_usage = (np.ceil(server_compute_capability / p_m)) * r_m  # 服务器的计算资源使用情况

        # 计算每个服务器上部署的服务实例数量
        service_instances = server_compute_resource_usage / r_m

        # 计算总成本和分项成本
        total_cost, cost_details = calculate_total_cost(result, m_edge, cost_edge, cost_cloud, p_user)

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
            f.write(f"Best Fitness (Jain Fairness Index): {F_jain:.4f}\n")

            f.write(f"Average Response Time of Best Solution: {avg_response_time:.2f} ms\n\n")

            f.write(f"Response Time Statistics by Priority:\n")
            for level, stats in response_stats.items():
                status = "OK" if stats["mean"] <= T_max[level] else "EXCEEDED"
                f.write(f"  Priority {level}: Mean={stats['mean']:.2f} ms (Limit: {T_max[level]} ms) [{status}]\n")

            f.write("\nUser-to-Server Assignment:\n")
            for i in range(len(user_positions)):
                server_idx = np.argmax(result[i])
                server_type = "Edge" if server_idx < m_edge else "Cloud"
                f.write(f"  User {i} -> Server {server_idx} ({server_type})\n")

            # 添加运行时间到实验结果文件
            end_time = time.time()
            execution_time = end_time - start_time
            f.write(f"\nTotal execution time: {execution_time:.2f} seconds\n")

        # ========== 可视化 ==========
        # 1. 绘制适应度变化曲线
        # plot_fitness_history(fitness_history, output_folder)

        # 2. 绘制响应时间分布
        plot_response_time_distribution(response_times, priorities, output_folder)

        # 3. 绘制平均响应时间柱状图
        plot_avg_response_time(response_times, priorities, output_folder, T_max)

        # 4. 绘制服务器资源使用情况
        plot_server_resource_usage(server_compute_resource_usage, R_edge, m_edge, output_folder)

        # 5. 绘制用户和服务器的连接图
        plot_user_server_connections(user_positions, server_positions, result, priorities, m_edge, output_folder)

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



