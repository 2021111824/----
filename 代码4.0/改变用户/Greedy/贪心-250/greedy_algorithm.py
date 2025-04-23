import math
import numpy as np
from calculations import calculate_weighted_jain_index
from constraints import check_constraints


# 贪心算法优化
def greedy_algorithm(n, m_edge, m_cloud, priorities, weights, cost_edge, cost_cloud, max_cost, T_max,
                     R_bandwidth, t_delay_e, t_delay_c, p_m, r_m, R_edge, user_data, p_user, P_allocation, P_cloud):
    """
    贪心算法
    - 按用户优先级顺序依次分配服务器
    - 选择 使 Jain 指数最大化 的服务器
    """
    n_users = n
    n_servers = m_edge + m_cloud
    individual = np.zeros((n_users, n_servers))

    valid_individual = False
    attempt_count = 0
    best_jain = -1

    while not valid_individual:
        individual = np.zeros((n_users, n_servers), dtype=int)
        server_compute_capability = np.zeros(n_servers)

        # 遍历用户，为每个用户寻找最优服务器
        for i in range(n):
            best_server = -1
            best_local_jain = -1

            # 优化：仅选择资源足够的服务器
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

                if jain_index > best_local_jain:
                    best_local_jain = jain_index
                    best_server = server_idx

            if best_server != -1:
                individual[i, best_server] = 1
                server_compute_capability[best_server] += P_allocation[i]

        # 检查约束
        valid_individual = check_constraints(individual, n, m_edge, m_cloud, priorities, R_bandwidth,
                                             cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                             user_data, p_user, P_allocation, p_m, r_m, R_edge, P_cloud)

        attempt_count += 1

        if attempt_count > 100:
            print("Warning: Too many attempts to generate valid individual. Moving forward.")
            break

    # 计算最终的加权 Jain 指数
    best_jain = calculate_weighted_jain_index(individual, n, m_edge, m_cloud, weights,
                                              t_delay_e, t_delay_c, user_data, R_bandwidth,
                                              p_user, P_allocation, P_cloud)

    return individual, best_jain
