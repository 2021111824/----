import numpy as np
import random
from calculations import assign_computational_capacity, compute_response_time
from constraints import check_constraints


def calculate_weighted_jain_index(individual, n, m_edge, m_cloud, compute_demands, request_sizes, weights,
                                  bandwidth_demands, P_edge, P_cloud, t_delay_e, t_delay_c):
    """
    计算加权 Jain 公平性指数
    """
    weighted_response_times = []

    user_capacities = assign_computational_capacity(individual, n, m_edge, m_cloud, compute_demands, P_edge, P_cloud)

    for i in range(n):
        server_idx = np.argmax(individual[i])
        is_edge = server_idx < m_edge
        response_time = compute_response_time(t_delay_e, t_delay_c, is_edge, request_sizes[i], user_capacities[i], bandwidth_demands[i])
        weighted_response_time = response_time * weights[i]
        weighted_response_times.append(weighted_response_time)

    weighted_response_times = np.array(weighted_response_times)

    numerator = np.sum(weighted_response_times) ** 2
    denominator = n * np.sum(weighted_response_times ** 2)
    return numerator / denominator if denominator != 0 else 0


# 贪心算法优化
def greedy_algorithm(n, m_edge, m_cloud, request_sizes, priorities, weights, compute_demands,
                     bandwidth_demands, P_edge, P_cloud, cost_edge,
                     cost_cloud, max_cost, T_max, R_compute, R_bandwidth, t_delay_e, t_delay_c):
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
        server_resources = np.zeros((n_servers, 2))  # COMPUTE, BANDWIDTH

        # 遍历用户，为每个用户寻找最优服务器
        for i in sorted_indices:
            best_server = -1
            best_jain = -1

            # 优化：同时计算多个服务器的可能性--仅选择资源足够的服务器
            potential_servers = [
                j for j in range(n_servers) if
                (server_resources[j, 0] + compute_demands[i] <= R_compute[j] and
                 server_resources[j, 1] + bandwidth_demands[i] <= R_bandwidth[j])
            ]

            # 计算 加权Jain指数，选择最优服务器
            for server_idx in potential_servers:
                temp_individual = individual.copy()
                temp_individual[i, server_idx] = 1

                jain_index = calculate_weighted_jain_index(temp_individual, n, m_edge, m_cloud, compute_demands,
                                                           request_sizes, weights, bandwidth_demands,
                                                           P_edge, P_cloud, t_delay_e, t_delay_c)

                if jain_index > best_jain:
                    best_jain = jain_index
                    best_server = server_idx
                    print(best_jain)

            if best_server != -1:
                individual[i, best_server] = 1
                server_resources[best_server, 0] += compute_demands[i]
                server_resources[best_server, 1] += bandwidth_demands[i]

        # 检查约束
        valid_individual = check_constraints(individual, n, m_edge, m_cloud, priorities,
                                             R_compute, R_bandwidth, compute_demands, bandwidth_demands,
                                             cost_edge, cost_cloud, max_cost, T_max, request_sizes,
                                             P_edge, P_cloud, t_delay_e, t_delay_c)

        attempt_count += 1

        if attempt_count > 100:
            print("Warning: Too many attempts to generate valid individual. Moving forward.")
            break

    # 贪心后进入局部优化
    individual = local_optimization(individual, n, m_edge, m_cloud, request_sizes, priorities,
                                    weights, P_edge, P_cloud,
                                    R_compute, R_bandwidth, compute_demands, bandwidth_demands,
                                    cost_edge, cost_cloud, max_cost, T_max,
                                    t_delay_e, t_delay_c)
    return individual


# 模拟退火（SA）局部优化
def local_optimization(individual, n, m_edge, m_cloud, request_sizes, priorities,
                       weights, P_edge, P_cloud,
                       R_compute, R_bandwidth, compute_demands, bandwidth_demands,
                       cost_edge, cost_cloud, max_cost, T_max,
                       t_delay_e, t_delay_c,
                       max_iters=100, initial_temp=100, alpha=0.99):
    """
    模拟退火 + 邻域搜索优化（加入约束检查）
    目的：优化贪心算法的结果，跳出局部最优
    - 交换（swap）：两个用户交换服务器分配（如果满足约束）
    - 重新分配（reassign）：随机选择一个用户，重新分配到合适的服务器
    - 温度衰减：动态调整，避免陷入局部最优
    - 约束检查：确保优化后仍满足所有资源和分配约束

    参数：
    max_iters=100       迭代次数
    initial_temp=100    初始温度
    alpha=0.99          温度衰减率
    """
    num_users = n
    num_servers = m_edge + m_cloud
    previous_jain = calculate_weighted_jain_index(individual, n, m_edge, m_cloud, compute_demands, request_sizes,
                                                  weights, bandwidth_demands, P_edge, P_cloud, t_delay_e, t_delay_c)

    best_jain = previous_jain
    best_individual = individual.copy()

    temp = initial_temp  # 初始化温度

    # 循环优化
    for _ in range(max_iters):
        improved = False
        operation = random.choice(["swap", "reassign"])  # 随机选择优化方式
        new_individual = individual.copy()

        # 交换策略（swap）
        if operation == "swap":
            i, j = np.random.choice(num_users, 2, replace=False)
            if np.argmax(individual[i]) != np.argmax(individual[j]):  # 确保交换有意义
                new_individual[i, np.argmax(individual[j])] = 1
                new_individual[j, np.argmax(individual[i])] = 1
                new_individual[i, np.argmax(individual[i])] = 0
                new_individual[j, np.argmax(individual[j])] = 0

        # 重新分配策略（reassign）
        elif operation == "reassign":
            i = np.random.choice(num_users)
            current_server = np.argmax(individual[i])

            feasible_servers = [j for j in range(num_servers)
                                if check_constraints(new_individual, n, m_edge, m_cloud, priorities,
                                                     R_compute, R_bandwidth, compute_demands, bandwidth_demands,
                                                     cost_edge, cost_cloud, max_cost, T_max, request_sizes,
                                                     P_edge, P_cloud, t_delay_e, t_delay_c)]
            if feasible_servers:
                new_server = random.choice(feasible_servers)
                new_individual[i, current_server] = 0
                new_individual[i, new_server] = 1

        # 检查约束是否满足
        valid_individual = check_constraints(new_individual, n, m_edge, m_cloud, priorities,
                                             R_compute, R_bandwidth, compute_demands, bandwidth_demands,
                                             cost_edge, cost_cloud, max_cost, T_max, request_sizes,
                                             P_edge, P_cloud, t_delay_e, t_delay_c)
        if not valid_individual:
            continue  # 如果新解不满足约束，则跳过此次优化

        # 计算新 Jain 指数
        new_jain = calculate_weighted_jain_index(new_individual, n, m_edge, m_cloud, compute_demands, request_sizes,
                                                 weights, bandwidth_demands, P_edge, P_cloud, t_delay_e, t_delay_c)

        # 接受新解的条件
        delta = new_jain - previous_jain  # 计算能量变化
        # 计算接受概率
        # delta > 0 ；必然接受
        # delta < 0 ；仍然有概率 exp(delta / temp) 接受。
        # 最终决定：
        # 若 delta > 一定阈值，直接接受；
        # 若 prob_accept > np.random.rand()，接受；
        # 否则，拒绝新解。
        prob_accept = np.exp(max(delta / temp, -500))

        if delta > (1e-3 * temp / initial_temp) or prob_accept > np.random.rand():
            individual = new_individual
            previous_jain = new_jain
            improved = True

        if new_jain > best_jain:
            best_jain = new_jain
            best_individual = new_individual.copy()

        temp *= alpha  # 温度衰减

        if not improved:
            break

    return best_individual