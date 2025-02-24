# 主程序模块--main.py
import os
import numpy as np

from calculations import calculate_response_stats, calculate_total_cost, assign_computational_capacity, \
    compute_response_time
from greedy_algorithm import greedy_algorithm, calculate_weighted_jain_index
from initialize import initialize_topology
from visualization import save_priority_distribution, plot_user_server_distribution, plot_fitness_history, \
    plot_response_time_distribution, plot_avg_response_time, plot_server_resource_usage, plot_user_server_connections, \
    plot_cost_distribution

# ========== 主程序入口 ==========
if __name__ == "__main__":

    # ========== 数据初始化 ==========
    n, m_edge, m_cloud, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, \
        T_max, cost_edge, cost_cloud, p_net, max_cost, \
        user_positions, request_sizes, priorities, weights, server_positions, \
        R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands = initialize_topology()

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
    result = greedy_algorithm(user_positions, server_positions, request_sizes, priorities, weights, cpu_demands,
                              mem_demands, bandwidth_demands, m_edge, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud,
                              cost_edge, cost_cloud, p_net, max_cost, T_max, R_cpu, R_mem, R_bandwidth)

    # ========== 结果分析 ==========
    if result is None:
        print("未找到满足约束条件的分配方案。")
    else:
        response_times = []
        user_capacities = assign_computational_capacity(result, user_positions, server_positions, request_sizes,
                                                        P_edge, P_cloud, m_edge, priorities)
        for i in range(len(user_positions)):
            server_idx = np.argmax(result[i])
            is_edge = server_idx < m_edge
            response_time = compute_response_time(user_positions[i], server_positions[server_idx], is_edge,
                                                  request_sizes[i], user_capacities[i], v_edge, v_cloud, b_edge,
                                                  b_cloud)
            response_times.append(response_time)

        # 平均响应时间
        avg_response_time = np.mean(response_times)

        # 响应时间统计
        response_stats = calculate_response_stats(response_times, priorities)

        # 计算加权Jain公平性指数
        F_jain = calculate_weighted_jain_index(result, user_positions, server_positions, request_sizes, priorities, weights,
                                                    m_edge, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud)

        # 记录资源使用情况
        server_cpu_usage = np.zeros(len(server_positions))
        server_mem_usage = np.zeros(len(server_positions))
        server_bandwidth_usage = np.zeros(len(server_positions))
        request_sizes_per_server = np.zeros(len(server_positions))

        # 遍历用户，统计每台服务器的资源使用
        for i in range(len(user_positions)):
            server_idx = np.argmax(result[i])  # 获取用户分配到的服务器
            is_edge = server_idx < m_edge  # 判断是否是边缘服务器

            # 使用用户的实际资源需求更新服务器资源使用
            server_cpu_usage[server_idx] += cpu_demands[i]  # 用户的 CPU 需求
            server_mem_usage[server_idx] += mem_demands[i]  # 用户的内存需求
            server_bandwidth_usage[server_idx] += bandwidth_demands[i]  # 用户的带宽需求

        # 计算总成本和分项成本
        total_cost, cost_details = calculate_total_cost(
            result,  # 用户到服务器的分配矩阵
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

        print(f"Simulation results saved to '{output_folder}'.")

        # ========== 可视化 ==========
        # 1. 绘制适应度变化曲线
        # plot_fitness_history(fitness_history, output_folder)

        # 2. 绘制响应时间分布
        plot_response_time_distribution(response_times, priorities, output_folder)

        # 3. 绘制平均响应时间柱状图
        plot_avg_response_time(response_times, priorities, output_folder, T_max)

        # 4. 绘制服务器资源使用情况
        plot_server_resource_usage(server_cpu_usage, server_mem_usage, server_bandwidth_usage,
                                   R_cpu, R_mem, R_bandwidth, m_edge, output_folder)

        # 5. 绘制用户和服务器的连接图
        plot_user_server_connections(user_positions, server_positions, result, priorities, m_edge, output_folder)

        # 6. 绘制服务器部署成本图
        plot_cost_distribution(cost_details, output_folder,
                               total_edge_cost=cost_details['edge']['fixed'] + cost_details['edge']['cpu'] +
                                               cost_details['edge']['mem'] + cost_details['edge']['bandwidth'],
                               total_cloud_cost=cost_details['cloud']['cpu'] + cost_details['cloud']['mem'] +
                                                cost_details['cloud']['bandwidth'] + cost_details['cloud']['network'],
                               total_cost=total_cost,
                               cost_limit=max_cost)
