# 修复模块--repair.py
import numpy as np
from calculations import compute_response_time, calculate_response_stats, assign_computational_capacity  # 引入响应时间计算函数


def repair_individual(individual, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                      cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                      request_sizes, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge):
    """
    修复不满足约束的个体，确保每个用户都最终有服务器分配
    """
    n_users, n_servers = individual.shape

    # 计算每个用户的计算能力
    user_capacities = assign_computational_capacity(user_positions, server_positions, request_sizes, P_edge, P_cloud,
                                                    m_edge, priorities)

    # 初始化资源使用情况
    server_cpu_usage = np.zeros(n_servers)
    server_mem_usage = np.zeros(n_servers)
    server_bandwidth_usage = np.zeros(n_servers)

    # 更新资源使用情况
    for i in range(n_users):
        server_idx = np.argmax(individual[i])  # 当前分配的服务器
        server_cpu_usage[server_idx] += cpu_demands[i]
        server_mem_usage[server_idx] += mem_demands[i]
        server_bandwidth_usage[server_idx] += bandwidth_demands[i]

    # 计算每个用户的响应时间
    response_times = [
        compute_response_time(
            user_positions[i], server_positions[np.argmax(individual[i])],
            np.argmax(individual[i]) < m_edge, request_sizes[i], user_capacities[i],
            v_edge, v_cloud, b_edge, b_cloud
        ) for i in range(n_users)
    ]

    # 持续修复，直到满足所有约束
    while True:
        # 记录是否进行了修复
        any_repair = False

        # 检查每个用户是否满足约束
        for i in range(n_users):
            server_idx = np.argmax(individual[i])  # 当前分配的服务器
            priority = priorities[i]

            # 计算该优先级的所有用户的平均响应时间
            avg_response_time = np.mean([response_times[j] for j in range(n_users) if priorities[j] == priority])

            # 检查该优先级的平均响应时间和资源约束是否超限
            if (server_cpu_usage[server_idx] > R_cpu[server_idx] or
                    server_mem_usage[server_idx] > R_mem[server_idx] or
                    server_bandwidth_usage[server_idx] > R_bandwidth[server_idx] or
                    avg_response_time > T_max[priority]):

                # 清除当前分配
                individual[i, server_idx] = 0
                server_cpu_usage[server_idx] -= cpu_demands[i]
                server_mem_usage[server_idx] -= mem_demands[i]
                server_bandwidth_usage[server_idx] -= bandwidth_demands[i]

                # 尝试重新分配到符合资源和响应时间限制的服务器
                valid_servers = [
                    new_server_idx for new_server_idx in range(n_servers)
                    if (server_cpu_usage[new_server_idx] + cpu_demands[i] <= R_cpu[new_server_idx] and
                        server_mem_usage[new_server_idx] + mem_demands[i] <= R_mem[new_server_idx] and
                        server_bandwidth_usage[new_server_idx] + bandwidth_demands[i] <= R_bandwidth[new_server_idx])
                ]

                # 随机从符合条件的服务器中选择一个
                if valid_servers:
                    new_server_idx = np.random.choice(valid_servers)

                    # 更新分配
                    individual[i, new_server_idx] = 1
                    server_cpu_usage[new_server_idx] += cpu_demands[i]
                    server_mem_usage[new_server_idx] += mem_demands[i]
                    server_bandwidth_usage[new_server_idx] += bandwidth_demands[i]

                    # 重新计算当前用户的响应时间
                    response_times[i] = compute_response_time(
                        user_positions[i], server_positions[new_server_idx],
                        new_server_idx < m_edge, request_sizes[i], user_capacities[i],
                        v_edge, v_cloud, b_edge, b_cloud
                    )

                    any_repair = True
                else:
                    # 如果没有有效的服务器，跳过当前用户
                    continue

        # 如果没有进行任何修复，跳出循环
        if not any_repair:
            break

    # 重新计算每个优先级的平均响应时间
    response_stats = calculate_response_stats(response_times, priorities)

    # 如果平均响应时间没有超限，退出修复
    for priority, stats in response_stats.items():
        avg_response_time = stats["mean"]
        if avg_response_time > T_max[priority]:
            # 如果修复后仍未满足约束，则继续修复
            return repair_individual(individual, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                                     cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                                     request_sizes, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)

    return individual  # 返回修复后的个体
