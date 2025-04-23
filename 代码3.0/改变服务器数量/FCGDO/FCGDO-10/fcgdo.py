import numpy as np
from calculations import compute_response_times, calculate_weighted_jain_index
from constraints import check_constraints


# fcgdo 算法优化
def fcgdo_algorithm(n, m_edge, m_cloud, priorities, weights, cost_edge, cost_cloud, max_cost, T_max,
                    R_bandwidth, t_delay_e, t_delay_c, p_m, r_m, R_edge, user_data, p_user, P_allocation, P_cloud):
    n_users = n
    n_servers = m_edge + m_cloud
    individual = np.zeros((n_users, n_servers), dtype=int)

    # 按优先级降序排列用户
    sorted_indices = np.argsort(priorities)[::-1]

    valid_individual = False
    attempt_count = 0
    bad_connections = []
    no_improvement_count = 0
    max_no_improvement = 1
    max_iterations = 3
    iteration_count = 0

    while not valid_individual:
        individual = np.zeros((n_users, n_servers), dtype=int)
        server_compute_capability = np.zeros(n_servers)
        # 缓存初始的Jain指数
        current_jain = calculate_weighted_jain_index(individual, n, m_edge, m_cloud, weights, t_delay_e, t_delay_c,
                                                     user_data, R_bandwidth, p_user, P_allocation, P_cloud)

        for i in sorted_indices:
            best_server = -1
            best_jain = -1

            # 计算潜在服务器
            potential_servers = np.where(
                np.ceil((server_compute_capability + P_allocation[i]) / p_m) * r_m <= R_edge
            )[0]

            # 预计算部分数据
            temp_individual_base = individual.copy()
            temp_individual_base[i] = 0

            for server_idx in potential_servers:
                temp_individual = temp_individual_base.copy()
                temp_individual[i, server_idx] = 1

                # 计算加权Jain指数
                jain_index = calculate_weighted_jain_index(temp_individual, n, m_edge, m_cloud, weights,
                                                           t_delay_e, t_delay_c, user_data, R_bandwidth,
                                                           p_user, P_allocation, P_cloud)

                if jain_index > best_jain:
                    best_jain = jain_index
                    best_server = server_idx

            if best_server != -1:
                individual[i, best_server] = 1
                server_compute_capability[best_server] += P_allocation[i]
                new_jain = calculate_weighted_jain_index(individual, n, m_edge, m_cloud, weights, t_delay_e, t_delay_c,
                                                         user_data, R_bandwidth, p_user, P_allocation, P_cloud)
                if new_jain < current_jain:
                    bad_connections.append(i)
                current_jain = new_jain

        # 检查约束
        valid_individual = check_constraints(individual, n, m_edge, m_cloud, priorities, R_bandwidth,
                                             cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                             user_data, p_user, P_allocation, p_m, r_m, R_edge, P_cloud)

        attempt_count += 1

        if attempt_count > 100:
            print("Warning: Too many attempts to generate valid individual. Moving forward.")
            break

    # 对 bad_connections 中的请求进行迁移优化
    individual, current_jain = migrate_requests(individual, n, m_edge, m_cloud, weights, priorities, R_bandwidth,
                                                cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                                user_data, p_user, P_allocation, p_m, r_m, R_edge, bad_connections, P_cloud)

    # 多次迁移整体请求以提升优化效果
    while True:
        all_users = set(range(n))
        individual, new_jain = migrate_requests(individual, n, m_edge, m_cloud, weights, priorities, R_bandwidth,
                                                cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                                user_data, p_user, P_allocation, p_m, r_m, R_edge, all_users, P_cloud)

        if new_jain <= current_jain:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
            current_jain = new_jain

        if no_improvement_count >= max_no_improvement or iteration_count >= max_iterations:
            break

        iteration_count += 1

    response_times = compute_response_times(individual, n, m_edge, m_cloud, t_delay_e, t_delay_c,
                                            user_data, R_bandwidth, p_user, P_allocation, P_cloud)

    return individual, current_jain, response_times


# 迁移请求
def migrate_requests(individual, n, m_edge, m_cloud, weights, priorities, R_bandwidth, cost_edge, cost_cloud, max_cost,
                     T_max, t_delay_e, t_delay_c, user_data, p_user, P_allocation, p_m, r_m, R_edge, user_indices, P_cloud):
    current_jain = calculate_weighted_jain_index(individual, n, m_edge, m_cloud, weights, t_delay_e, t_delay_c,
                                                 user_data, R_bandwidth, p_user, P_allocation, P_cloud)
    n_servers = m_edge + m_cloud

    for user_idx in user_indices:
        best_server = -1
        max_jain = -1
        current_server = np.argmax(individual[user_idx])

        # 预计算部分数据
        temp_individual_base = individual.copy()
        temp_individual_base[user_idx] = 0

        for server_idx in range(n_servers):
            if server_idx == current_server:
                continue
            temp_individual = temp_individual_base.copy()
            temp_individual[user_idx, server_idx] = 1

            # 检查约束
            valid_individual = check_constraints(temp_individual, n, m_edge, m_cloud, priorities, R_bandwidth,
                                                 cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                                 user_data, p_user, P_allocation, p_m, r_m, R_edge, P_cloud)
            if not valid_individual:
                continue

            new_jain = calculate_weighted_jain_index(temp_individual, n, m_edge, m_cloud, weights, t_delay_e, t_delay_c,
                                                     user_data, R_bandwidth, p_user, P_allocation, P_cloud)
            if new_jain > max_jain:
                max_jain = new_jain
                best_server = server_idx

        if best_server != -1 and max_jain > current_jain:
            individual[user_idx, current_server] = 0
            individual[user_idx, best_server] = 1
            current_jain = max_jain

    return individual, current_jain
