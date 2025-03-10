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
    for i in range(n_users):  # 对每个用户计算加权响应时间
        server_idx = np.argmax(individual[i])  # 找到该用户分配的服务器索引
        is_edge = server_idx < m_edge
        response_time = compute_response_time(user_positions[i], server_positions[server_idx], is_edge,
                                              request_sizes[i], user_capacities[i], v_edge, v_cloud, b_edge, b_cloud)
        weighted_response_time = response_time * weights[i]
        weighted_response_times.append(weighted_response_time)
    weighted_response_times = np.array(weighted_response_times)

    numerator = np.sum(weighted_response_times) ** 2  # 加权Jain公平性指数的分子
    denominator = n_users * np.sum(weighted_response_times ** 2)  # 加权Jain公平性指数的分母
    return numerator / denominator if denominator != 0 else 0  # 返回加权Jain公平性指数


# 贪心算法
def greedy_algorithm(user_positions, server_positions, request_sizes, priorities, weights, cpu_demands, mem_demands,
                     bandwidth_demands, m_edge, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, cost_edge, cost_cloud,
                     p_net, max_cost, T_max, R_cpu, R_mem, R_bandwidth):

    n_users = len(user_positions)
    n_servers = len(server_positions)
    individual = np.zeros((n_users, n_servers))  # 初始化用户到服务器的分配矩阵，全为0

    # 优先级排序，最后返回由高到低排列的用户的索引
    # 优先分配高优先级用户
    sorted_indices = np.argsort(priorities)[::-1]

    valid_individual = False  # 初始时，分配是无效的
    attempt_count = 0  # 尝试计数--用来限制尝试次数，防止死循环。

    while not valid_individual:
        individual = np.zeros((n_users, n_servers), dtype=int)  # 每次尝试时重新初始化分配矩阵
        # 初始化每个服务器的资源使用情况
        server_cpu_usage = np.zeros(n_servers)
        server_mem_usage = np.zeros(n_servers)
        server_bandwidth_usage = np.zeros(n_servers)

        # 依次为每个用户选择最佳服务器，目标是使加权Jain最大
        for i in sorted_indices:  # 遍历每个用户，按照优先级顺序
            best_jain = -1  # 初始化最佳加权Jain
            best_server = -1  # 初始化最佳服务器索引
            for server_idx in range(n_servers):  # 遍历所有服务器
                # 临时分配用户到当前服务器
                temp_individual = individual.copy()  # 复制当前的分配矩阵
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

                    if jain_index > best_jain:  # 如果当前分配的Jain指数更优
                        best_jain = jain_index  # 更新最佳Jain指数
                        best_server = server_idx   # 记录最佳服务器
                        print("当前的最佳")
                        print(best_jain)

            if best_server != -1:  # 如果找到了最佳服务器
                # 分配用户到使 Jain 公平性指数最大的服务器
                individual[i, best_server] = 1
                # 更新该服务器的资源使用情况
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

