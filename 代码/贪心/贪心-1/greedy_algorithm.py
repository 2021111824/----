import numpy as np
from calculations import assign_computational_capacity, compute_response_time, calculate_total_cost, calculate_response_stats
from constraints import check_constraints


def calculate_weighted_jain_index(individual, user_positions, server_positions, request_sizes, priorities, weights,
                                  m_edge, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud):
    """
    计算加权 Jain 公平性指数
    """
    n_users = len(user_positions)
    weighted_response_times = []
    user_capacities = assign_computational_capacity(individual, user_positions, server_positions, request_sizes,
                                                    P_edge, P_cloud, m_edge, priorities)
    for i in range(n_users):
        server_idx = np.argmax(individual[i])
        is_edge = server_idx < m_edge
        response_time = compute_response_time(user_positions[i], server_positions[server_idx], is_edge,
                                              request_sizes[i], user_capacities[i], v_edge, v_cloud, b_edge, b_cloud)
        weighted_response_time = response_time * weights[i]
        weighted_response_times.append(weighted_response_time)
    weighted_response_times = np.array(weighted_response_times)
    numerator = np.sum(weighted_response_times) ** 2
    denominator = n_users * np.sum(weighted_response_times ** 2)
    return numerator / denominator if denominator != 0 else 0


def calculate_server_resources(individual, cpu_demands, mem_demands, bandwidth_demands, R_cpu, R_mem, R_bandwidth):
    """
    计算每个服务器的剩余资源，返回剩余的 CPU、内存和带宽资源。
    """
    n_users, n_servers = individual.shape  # 获取用户数量和服务器数量
    server_resources = np.zeros((3, n_servers))  # 0: CPU, 1: Memory, 2: Bandwidth

    for server_idx in range(n_servers):  # 遍历所有服务器
        total_cpu_usage = 0
        total_mem_usage = 0
        total_bandwidth_usage = 0

        # 遍历所有用户，检查每个用户是否连接到当前服务器
        for user_idx in range(n_users):
            if individual[user_idx, server_idx] == 1:  # 如果用户连接到当前服务器
                total_cpu_usage += cpu_demands[user_idx]
                total_mem_usage += mem_demands[user_idx]
                total_bandwidth_usage += bandwidth_demands[user_idx]

        # 更新当前服务器的资源剩余情况
        server_resources[0, server_idx] = R_cpu[server_idx] - total_cpu_usage
        server_resources[1, server_idx] = R_mem[server_idx] - total_mem_usage
        server_resources[2, server_idx] = R_bandwidth[server_idx] - total_bandwidth_usage

    return server_resources


def greedy_algorithm(user_positions, server_positions, request_sizes, priorities, weights, cpu_demands, mem_demands,
                     bandwidth_demands, m_edge, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, cost_edge, cost_cloud,
                     p_net, max_cost, T_max, R_cpu, R_mem, R_bandwidth):

    n_users = len(user_positions)
    n_servers = len(server_positions)
    individual = np.zeros((n_users, n_servers))  # 初始化用户到服务器的分配矩阵
    best_jain_index = 0
    best_individual = individual.copy()

    # 优先级排序，优先分配高优先级用户
    sorted_indices = np.argsort(priorities)[::-1]

    valid_individual = False
    attempt_count = 0  # 尝试计数
    while not valid_individual:
        individual = np.zeros((n_users, n_servers), dtype=int)
        server_cpu_usage = np.zeros(n_servers)
        server_mem_usage = np.zeros(n_servers)
        server_bandwidth_usage = np.zeros(n_servers)

        for i in sorted_indices:  # 遍历每个用户，按照优先级顺序
            best_jain = -1
            best_server = -1
            for server_idx in range(n_servers):  # 遍历所有服务器
                # 临时分配用户到当前服务器
                temp_individual = individual.copy()
                temp_individual[i, server_idx] = 1

                # 检查该服务器是否资源满足限制
                if (server_cpu_usage[server_idx] + cpu_demands[i] <= R_cpu[server_idx] and
                        server_mem_usage[server_idx] + mem_demands[i] <= R_mem[server_idx] and
                        server_bandwidth_usage[server_idx] + bandwidth_demands[i] <= R_bandwidth[server_idx]):
                    # 计算 Jain 公平性指数
                    jain_index = calculate_weighted_jain_index(temp_individual, user_positions, server_positions,
                                                               request_sizes, priorities, weights,
                                                               m_edge, v_edge, v_cloud, b_edge, b_cloud,
                                                               P_edge, P_cloud)
                    if jain_index > best_jain:
                        best_jain = jain_index
                        best_server = server_idx

            if best_server != -1:
                # 分配用户到使 Jain 公平性指数最大的服务器
                individual[i, best_server] = 1
                server_cpu_usage[best_server] += cpu_demands[i]
                server_mem_usage[best_server] += mem_demands[i]
                server_bandwidth_usage[best_server] += bandwidth_demands[i]

        # 检查是否满足所有约束
        valid_individual = check_constraints(individual, user_positions, server_positions, priorities,
                                             R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                                             cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                                             v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)

        attempt_count += 1

        if attempt_count > 100:  # 防止死循环，如果尝试次数过多则跳出
            print("Warning: Too many attempts to generate valid individual. Moving forward.")
            break

    return individual

