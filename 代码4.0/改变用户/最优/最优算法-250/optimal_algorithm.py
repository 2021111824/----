import math
import numpy as np
from calculations import calculate_weighted_jain_index
from constraints import check_constraints


# 基于最大化加权Jain指数的算法
def optimal_algorithm(n, m_edge, m_cloud, priorities, weights, cost_edge, cost_cloud, max_cost, T_max,
                      R_bandwidth, t_delay_e, t_delay_c, p_m, r_m, R_edge, user_data, p_user, P_allocation, P_cloud):
    n_users = n
    n_servers = m_edge + m_cloud
    best_individual = np.zeros((n_users, n_servers), dtype=int)
    best_jain = -1
    max_attempts = 10  # 最大尝试次数
    attempt_count = 0

    while attempt_count < max_attempts:
        individual = np.zeros((n_users, n_servers), dtype=int)
        server_compute_capability = np.zeros(n_servers)

        # 遍历用户，为每个用户寻找最优服务器
        for i in range(n):
            best_server_local = -1
            best_jain_local = -1
            # 仅选择资源足够的服务器
            potential_servers = [
                j for j in range(n_servers) if
                math.ceil((server_compute_capability[j] + P_allocation[i]) / p_m) * r_m <= R_edge[j]
            ]
            for server_idx in potential_servers:
                temp_individual = individual.copy()
                temp_individual[i, server_idx] = 1
                jain_index = calculate_weighted_jain_index(temp_individual, n, m_edge, m_cloud, weights,
                                                           t_delay_e, t_delay_c, user_data, R_bandwidth,
                                                           p_user, P_allocation, P_cloud)
                if jain_index > best_jain_local:
                    best_jain_local = jain_index
                    best_server_local = server_idx
            if best_server_local != -1:
                individual[i, best_server_local] = 1
                server_compute_capability[best_server_local] += P_allocation[i]

        # 检查约束
        if check_constraints(individual, n, m_edge, m_cloud, priorities, R_bandwidth,
                             cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                             user_data, p_user, P_allocation, p_m, r_m, R_edge, P_cloud):
            current_jain = calculate_weighted_jain_index(individual, n, m_edge, m_cloud, weights,
                                                         t_delay_e, t_delay_c, user_data, R_bandwidth,
                                                         p_user, P_allocation, P_cloud)
            if current_jain > best_jain:
                best_jain = current_jain
                best_individual = individual.copy()
        attempt_count += 1

    return best_individual
