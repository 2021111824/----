import math
import numpy as np
from calculations import assign_bandwidth_capacity, compute_response_time
from constraints import check_constraints
import random

# 固定 numpy 的随机种子
np.random.seed(60)
# 固定 random 模块的随机种子
random.seed(60)


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
        response_time = compute_response_time(t_delay_e[i][server_idx], t_delay_c[i], is_edge, user_data[i],
                                              user_bandwidth[i], p_user[i], P_allocation[i])
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
                    # print(best_jain)

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

    # 贪心后进入局部优化
    best_individual, best_jain, best_response_times = sa_local_optimization(individual, n, m_edge, m_cloud, priorities,
                                                                            weights,
                                                                            R_bandwidth, cost_edge, cost_cloud,
                                                                            max_cost, T_max,
                                                                            t_delay_e, t_delay_c, user_data, p_user,
                                                                            P_allocation, p_m, r_m, R_edge)
    return best_individual, best_jain, best_response_times


# 模拟退火（SA）局部优化
def sa_local_optimization(individual, n, m_edge, m_cloud, priorities, weights,
                          R_bandwidth, cost_edge, cost_cloud, max_cost, T_max,
                          t_delay_e, t_delay_c, user_data, p_user, P_allocation, p_m, r_m, R_edge,
                          max_iters=100, initial_temp=100, alpha=0.99):
    """
    模拟退火 + 邻域搜索优化（加入约束检查）
    目的：优化贪心算法的结果，跳出局部最优
    - 交换（swap）：两个用户交换服务器分配（如果满足约束）
    - 重新分配（reassign）：随机选择一个用户，重新分配到合适的服务器
    - 挪动（move_user）：随机选择一个用户，将其移动到其他服务器
    - 子集重新分配（reassign_subset）：重新分配一组用户
    - 温度衰减：动态调整，避免陷入局部最优
    - 约束检查：确保优化后仍满足所有资源和分配约束

    参数：
    max_iters=100       迭代次数
    initial_temp=100    初始温度
    alpha=0.99          温度衰减率
    """
    num_users = n
    num_servers = m_edge + m_cloud
    previous_jain = calculate_weighted_jain_index(individual, n, m_edge, m_cloud, weights, t_delay_e, t_delay_c,
                                                  user_data, R_bandwidth, p_user, P_allocation)

    best_jain = previous_jain
    best_individual = individual.copy()
    best_response_times = compute_response_times(individual, n, m_edge, m_cloud, t_delay_e, t_delay_c,
                                                 user_data, R_bandwidth, p_user, P_allocation)

    temp = initial_temp  # 初始化温度

    # 循环优化
    for _ in range(max_iters):
        improved = False
        operation = random.choice(["swap", "reassign", "move_user", "reassign_subset"])  # 随机选择优化方式
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
                                if check_constraints(new_individual, n, m_edge, m_cloud, priorities, R_bandwidth,
                                                     cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                                     user_data, p_user, P_allocation, p_m, r_m, R_edge)]
            if feasible_servers:
                new_server = random.choice(feasible_servers)
                new_individual[i, current_server] = 0
                new_individual[i, new_server] = 1

        # 挪动策略（move_user）
        elif operation == "move_user":
            i = np.random.choice(num_users)  # 随机选择一个用户
            current_server = np.argmax(individual[i])

            feasible_servers = [j for j in range(num_servers) if j != current_server]
            if feasible_servers:
                new_server = random.choice(feasible_servers)
                new_individual[i, current_server] = 0
                new_individual[i, new_server] = 1

        # 子集重新分配策略（reassign_subset）
        elif operation == "reassign_subset":
            # 随机选择一组用户
            subset_size = np.random.randint(1, num_users // 2)  # 随机选择子集大小
            subset_users = np.random.choice(num_users, subset_size, replace=False)
            for i in subset_users:
                current_server = np.argmax(individual[i])
                feasible_servers = [j for j in range(num_servers) if j != current_server]
                if feasible_servers:
                    new_server = random.choice(feasible_servers)
                    new_individual[i, current_server] = 0
                    new_individual[i, new_server] = 1

        # 检查约束是否满足
        valid_individual = check_constraints(new_individual, n, m_edge, m_cloud, priorities, R_bandwidth,
                                             cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                             user_data, p_user, P_allocation, p_m, r_m, R_edge)
        if not valid_individual:
            continue  # 如果新解不满足约束，则跳过此次优化

        # 计算新 Jain 指数
        new_jain = calculate_weighted_jain_index(individual, n, m_edge, m_cloud, weights, t_delay_e, t_delay_c,
                                                 user_data, R_bandwidth, p_user, P_allocation)

        # 接受新解的条件
        delta = new_jain - previous_jain  # 计算能量变化
        # 计算接受概率
        prob_accept = np.exp(max(delta / temp, -500))

        if delta > (1e-3 * temp / initial_temp) or prob_accept > np.random.rand():
            individual = new_individual
            previous_jain = new_jain
            improved = True

        if new_jain > best_jain:
            best_jain = new_jain
            best_individual = new_individual.copy()
            best_response_times = compute_response_times(best_individual, n, m_edge, m_cloud, t_delay_e, t_delay_c,
                                                         user_data, R_bandwidth, p_user, P_allocation)

        temp *= alpha  # 温度衰减

        if not improved:
            break

    return best_individual, best_jain, best_response_times  # 返回最优解的加权 Jain 指数和响应时间


# 计算响应时间
def compute_response_times(individual, n, m_edge, m_cloud, t_delay_e, t_delay_c, user_data, R_bandwidth, p_user,
                           P_allocation):
    """
    计算每个用户的响应时间
    """
    response_times = []
    user_bandwidth = assign_bandwidth_capacity(individual, n, m_edge, m_cloud, user_data, R_bandwidth)
    for i in range(n):
        server_idx = np.argmax(individual[i])
        is_edge = server_idx < m_edge
        response_time = compute_response_time(t_delay_e[i][server_idx], t_delay_c[i], is_edge, user_data[i],
                                              user_bandwidth[i], p_user[i], P_allocation[i])
        response_times.append(response_time)
    return np.array(response_times)
