# 修复模块--repair.py
import math

import numpy as np
from calculations import compute_response_time, calculate_response_stats, assign_bandwidth_capacity  # 引入响应时间计算函数


def repair_individual(individual, n, m_edge, m_cloud, user_data, R_bandwidth, priorities, T_max,
                      p_user, P_allocation, t_delay_e, t_delay_c, p_m, r_m, R_edge, P_cloud):
    """
    修复不满足约束的个体，确保每个用户都最终有服务器分配
    """
    n_users, n_servers = individual.shape

    # 根据分配情况计算每个用户的带宽
    user_bandwidth = assign_bandwidth_capacity(individual, n, m_edge, m_cloud, user_data, R_bandwidth)

    # 记录各用户响应时间
    response_times = []

    # 初始化资源使用情况
    server_compute_capability = np.zeros(n_servers)

    # 更新资源使用情况
    for i in range(n_users):
        server_idx = np.argmax(individual[i])  # 当前分配的服务器
        is_edge = server_idx < m_edge  # 是否为边缘服务器

        server_compute_capability[server_idx] += P_allocation[i]

        # 更新每个用户的响应时间
        response_time = compute_response_time(
            t_delay_e[i][server_idx], t_delay_c[i], is_edge, user_data[i], user_bandwidth[i], p_user[i],
            P_allocation[i], P_cloud)
        response_times.append(response_time)

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
            if (math.ceil((server_compute_capability[server_idx]) / p_m) * r_m > R_edge[server_idx] or
                    avg_response_time > T_max[priority]):

                # 清除当前分配
                individual[i, server_idx] = 0
                server_compute_capability[server_idx] -= P_allocation[i]

                # 尝试重新分配到符合资源和响应时间限制的服务器
                valid_servers = [
                    new_server_idx for new_server_idx in range(n_servers)
                    if (math.ceil((server_compute_capability[new_server_idx] + P_allocation[i]) / p_m) * r_m <= R_edge[new_server_idx])
                ]

                # 随机从符合条件的服务器中选择一个
                if valid_servers:
                    new_server_idx = np.random.choice(valid_servers)

                    # 更新分配
                    individual[i, new_server_idx] = 1
                    server_compute_capability[new_server_idx] += P_allocation[i]

                    # 重新计算计算能力
                    user_bandwidth = assign_bandwidth_capacity(individual, n, m_edge, m_cloud, user_data, R_bandwidth)

                    # 是否为边缘服务器
                    is_edge = new_server_idx < m_edge

                    # 重新计算当前用户的响应时间
                    response_times[i] = compute_response_time(
                        t_delay_e[i][new_server_idx], t_delay_c[i], is_edge,
                        user_data[i], user_bandwidth[i], p_user[i], P_allocation[i], P_cloud
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
            return repair_individual(individual, n, m_edge, m_cloud, user_data, R_bandwidth, priorities, T_max,
                                     p_user, P_allocation, t_delay_e, t_delay_c, p_m, r_m, R_edge, P_cloud)

    return individual  # 返回修复后的个体
