import math

import numpy as np
from constraints import check_constraints


# 随机算法
def random_algorithm(n, m_edge, m_cloud, priorities, weights, cost_edge, cost_cloud, max_cost, T_max,
                     R_bandwidth, t_delay_e, t_delay_c, p_m, r_m, R_edge, user_data, p_user, P_allocation, P_cloud):
    """
    随机算法
    - 随机选择服务器分配给用户
    """
    n_users = n
    n_servers = m_edge + m_cloud
    individual = np.zeros((n_users, n_servers))

    valid_individual = False
    attempt_count = 0

    while not valid_individual:
        individual = np.zeros((n_users, n_servers), dtype=int)
        server_compute_capability = np.zeros(n_servers)

        # 遍历用户，为每个用户随机选择一个服务器
        for i in range(n):
            # 随机选择一个可能的服务器
            potential_servers = [
                j for j in range(n_servers) if
                math.ceil((server_compute_capability[j] + P_allocation[i]) / p_m) * r_m <= R_edge[j]
            ]
            if potential_servers:
                selected_server = np.random.choice(potential_servers)
                individual[i, selected_server] = 1
                server_compute_capability[selected_server] += P_allocation[i]

        # 检查约束
        valid_individual = check_constraints(individual, n, m_edge, m_cloud, priorities, R_bandwidth,
                                             cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                             user_data, p_user, P_allocation, p_m, r_m, R_edge, P_cloud)

        attempt_count += 1

        if attempt_count > 100:
            print("Warning: Too many attempts to generate valid individual. Moving forward.")
            break

    return individual
