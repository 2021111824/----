import os
import numpy as np
import time

from calculations import calculate_response_stats, calculate_total_cost
from fcgdo import fcgdo_algorithm
from initialize import initialize_topology
from visualization import save_priority_distribution, plot_user_server_distribution,  \
    plot_response_time_distribution, plot_avg_response_time, plot_server_resource_usage, plot_user_server_connections, \
    plot_cost_distribution, plot_service_instance_distribution


# ========== 主程序入口 ==========
if __name__ == "__main__":

    # ========== 数据初始化 ==========
    n, user_positions, priorities, weights, user_data, p_user, P_allocation, T_max, \
        m_edge, m_cloud, server_positions, t_delay_e, t_delay_c, R_bandwidth, R_edge, P_edge, P_cloud, \
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

    # ========== 运行 FCGDO 算法 ==========
    # 记录算法开始时间
    start_time = time.time()

    result, best_jain, best_response_times = fcgdo_algorithm(n, m_edge, m_cloud, priorities, weights, cost_edge,
                                                             cost_cloud, max_cost, T_max, R_bandwidth, t_delay_e,
                                                             t_delay_c, p_m, r_m, R_edge, user_data, p_user,
                                                             P_allocation, P_cloud)

    # 记录算法结束时间
    end_time = time.time()

    # 算法总运行时间
    execution_time = end_time - start_time

    best_final_result = result
    best_final_jain = best_jain
    best_final_response_times = best_response_times

    # ========== 结果分析 ==========
    if best_final_result is None:
        print("未找到满足约束条件的分配方案。")
    else:
        # 响应时间赋值
        response_times = best_final_response_times

        # 平均响应时间
        avg_response_time = np.mean(best_final_response_times)

        # 响应时间统计
        response_stats = calculate_response_stats(best_final_response_times, priorities)

        # 计算加权Jain公平性指数
        F_jain = best_final_jain

        # 记录资源使用情况
        server_compute_capability = np.zeros(len(server_positions))

        # 向量化操作统计服务器计算能力需求
        server_indices = np.argmax(best_final_result, axis=1)
        np.add.at(server_compute_capability, server_indices, P_allocation)

        # 服务器上的计算资源使用情况
        server_compute_resource_usage = (np.ceil(server_compute_capability / p_m)) * r_m  # 服务器的计算资源使用情况

        # 计算每个服务器上部署的服务实例数量
        service_instances = server_compute_resource_usage / r_m

        # 计算总成本和分项成本
        total_cost, cost_details = calculate_total_cost(best_final_result, m_edge, cost_edge, cost_cloud, p_user)

        # 保存到文件
        cost_file_content = [
            "===== Cost Results =====",
            f"Total Cost: {total_cost:.2f}\n",
            "Cost Breakdown:",
            "  Edge Servers:"
        ]
        for key, value in cost_details["edge"].items():
            cost_file_content.append(f"    {key.capitalize()}: {value:.2f}")
        cost_file_content.extend([
            "  Cloud Servers:"
        ])
        for key, value in cost_details["cloud"].items():
            cost_file_content.append(f"    {key.capitalize()}: {value:.2f}")
        with open(os.path.join(output_folder, "cost_results.txt"), "w") as f:
            f.write("\n".join(cost_file_content))

        # 计算并保存每个服务器的服务实例数量到文件
        service_instance_file_content = [
            "===== Service Instances per Server ====="
        ]
        for j in range(len(server_positions)):
            server_type = "Edge" if j < m_edge else "Cloud"
            service_instance_file_content.append(f"Server {j} ({server_type}): {service_instances[j]} instances")
        with open(os.path.join(output_folder, "service_instances.txt"), "w") as f:
            f.write("\n".join(service_instance_file_content))

        print(f"Service instance results saved to '{output_folder}/service_instances.txt'.")

        # 保存详细结果到文件
        simulation_result_file_content = [
            "===== Simulation Results ====="
            
            f"\nBest Fitness (Jain Fairness Index): {F_jain:.4f}",

            f"\nTotal execution time: {execution_time:.2f} seconds",

            f"\nAverage Response Time of Best Solution: {avg_response_time:.2f} ms\n",
            "Response Time Statistics by Priority:"
        ]
        for level, stats in response_stats.items():
            status = "OK" if stats["mean"] <= T_max[level] else "EXCEEDED"
            simulation_result_file_content.append(f"  Priority {level}: Mean={stats['mean']:.2f} ms (Limit: {T_max[level]} ms) [{status}]")
        simulation_result_file_content.extend([
            "\nUser-to-Server Assignment:"
        ])
        for i in range(len(user_positions)):
            server_idx = np.argmax(best_final_result[i])
            server_type = "Edge" if server_idx < m_edge else "Cloud"
            simulation_result_file_content.append(f"  User {i} -> Server {server_idx} ({server_type})")

        with open(os.path.join(output_folder, "simulation_results.txt"), "w") as f:
            f.write("\n".join(simulation_result_file_content))

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
        plot_user_server_connections(user_positions, server_positions, best_final_result, priorities, m_edge, output_folder)

        # 6. 绘制服务器部署成本图
        plot_cost_distribution(cost_details, output_folder,
                               total_edge_cost=cost_details['edge']['fixed'],
                               total_cloud_cost=cost_details['cloud']['p_net'],
                               total_cost=total_cost,
                               cost_limit=max_cost)

        # 7. 绘制服务器上的服务实例部署情况
        plot_service_instance_distribution(service_instances, output_folder)


