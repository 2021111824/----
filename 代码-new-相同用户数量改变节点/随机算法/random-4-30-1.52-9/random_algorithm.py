import math

import numpy as np
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


# 随机算法
def random_algorithm(n, m_edge, m_cloud, priorities, weights, cost_edge, cost_cloud, max_cost, T_max,
                     R_bandwidth, t_delay_e, t_delay_c, p_m, r_m, R_edge, user_data, p_user, P_allocation):
    """
    随机算法
    - 随机选择服务器分配给用户
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

        # 遍历用户，为每个用户随机选择一个服务器
        for i in sorted_indices:
            # 随机选择一个可能的服务器
            potential_servers = [
                j for j in range(n_servers) if
                math.ceil((server_compute_capability[j] + p_user[i]) / p_m) * r_m <= R_edge[j]
            ]
            if potential_servers:
                selected_server = np.random.choice(potential_servers)
                individual[i, selected_server] = 1
                server_compute_capability[selected_server] += p_user[i]

        # 检查约束
        valid_individual = check_constraints(individual, n, m_edge, m_cloud, priorities, R_bandwidth,
                                             cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                             user_data, p_user, P_allocation, p_m, r_m, R_edge)

        attempt_count += 1

        if attempt_count > 100:
            print("Warning: Too many attempts to generate valid individual. Moving forward.")
            break

    return individual
