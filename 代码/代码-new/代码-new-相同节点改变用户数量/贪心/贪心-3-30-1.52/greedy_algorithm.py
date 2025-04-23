import math

import numpy as np
import random
from calculations import assign_bandwidth_capacity, compute_response_time
from constraints import check_constraints


def calculate_weighted_jain_index(individual, n, m_edge, m_cloud, weights, t_delay_e, t_delay_c, user_data, R_bandwidth,
                                  p_user, P_allocation):
    """
    计算加权 Jain 公平性指数
    """
    weighted_response_times = []

    user_bandwidth = assign_bandwidth_capacity(individual, n, m_edge, m_cloud, user_data, R_bandwidth)
    for i in range(n):
        server_idx = np.argmax(individual[i])
        is_edge = server_idx < m_edge
        response_time = compute_response_time(t_delay_e[i][server_idx], t_delay_c[i], is_edge, user_data[i], user_bandwidth[i], p_user[i], P_allocation[i])
        weighted_response_time = response_time * weights[i]
        weighted_response_times.append(weighted_response_time)

    weighted_response_times = np.array(weighted_response_times)

    numerator = np.sum(weighted_response_times) ** 2
    denominator = n * np.sum(weighted_response_times ** 2)
    return numerator / denominator if denominator != 0 else 0


# 贪心算法优化
def greedy_algorithm(n, m_edge, m_cloud, priorities, weights, cost_edge, cost_cloud, max_cost, T_max,
                     R_bandwidth, t_delay_e, t_delay_c, p_m, r_m, R_edge, user_data, p_user, P_allocation):
    """
    贪心算法
    - 按用户优先级顺序依次分配服务器
    - 选择 使 Jain 指数最大化 的服务器
    """
    n_users = n
    n_servers = m_edge + m_cloud
    individual = np.zeros((n_users, n_servers))

    # 优化：按优先级降序排列用户
    sorted_indices = np.argsort(priorities)[::-1]

    valid_individual = False
    attempt_count = 0

    while not valid_individual:
        individual = np.zeros((n_users, n_servers), dtype=int)
        server_compute_capability = np.zeros(n_servers)
        server_compute_resource_usage = np.zeros(n_servers)  # 边缘服务器的计算资源使用情况

        # 遍历用户，为每个用户寻找最优服务器
        for i in sorted_indices:
            best_server = -1
            best_jain = -1

            # 优化：同时计算多个服务器的可能性--仅选择资源足够的服务器
            potential_servers = [
                j for j in range(n_servers) if
                math.ceil((server_compute_capability[j] + p_user[i]) / p_m) * r_m <= R_edge[j]
            ]

            # 计算 加权Jain指数，选择最优服务器
            for server_idx in potential_servers:
                temp_individual = individual.copy()
                temp_individual[i, server_idx] = 1

                jain_index = calculate_weighted_jain_index(temp_individual, n, m_edge, m_cloud, weights,
                                                           t_delay_e, t_delay_c, user_data, R_bandwidth,
                                                           p_user, P_allocation)

                if jain_index > best_jain:
                    best_jain = jain_index
                    best_server = server_idx
                    print(best_jain)

            if best_server != -1:
                individual[i, best_server] = 1
                server_compute_capability[best_server] += p_user[i]

        # 检查约束
        valid_individual = check_constraints(individual, n, m_edge, m_cloud, priorities, R_bandwidth,
                                             cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                             user_data, p_user, P_allocation, p_m, r_m, R_edge)

        attempt_count += 1

        if attempt_count > 100:
            print("Warning: Too many attempts to generate valid individual. Moving forward.")
            break

    return individual
