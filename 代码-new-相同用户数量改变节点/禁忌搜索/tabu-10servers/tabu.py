import math
import numpy as np
from calculations import compute_response_time, assign_bandwidth_capacity
from constraints import check_constraints
from tqdm import tqdm
from repair import repair_individual


# 生成邻域解
def generate_neighbors(individual, m_edge, m_cloud):
    neighbors = []
    n_server = m_edge + m_cloud
    n = len(individual)
    for i in range(n):
        current_server = np.argmax(individual[i])  # 当前用户分配的服务器索引
        for j in range(n_server):
            if j != current_server:
                new_individual = individual.copy()
                new_individual[i] = np.zeros(n_server)
                new_individual[i, j] = 1
                neighbors.append(new_individual)
    return neighbors


# 计算适应度值，综合考虑公平性和约束条件
def compute_fitness(individual, n, m_edge, m_cloud, user_data, weights,
                    R_bandwidth, t_delay_e, t_delay_c, p_user, P_allocation):
    """
    计算适应度值，综合考虑公平性和约束条件。
    """
    # 根据分配情况计算每个用户的带宽
    user_bandwidth = assign_bandwidth_capacity(individual, n, m_edge, m_cloud, user_data, R_bandwidth)

    response_times = []

    for i in range(n):
        server_idx = np.argmax(individual[i])  # 用户分配到的服务器
        is_edge = server_idx < m_edge  # 是否为边缘服务器

        response_time = compute_response_time(
            t_delay_e[i][server_idx], t_delay_c[i], is_edge, user_data[i], user_bandwidth[i], p_user[i], P_allocation[i]
        )
        response_times.append(response_time)

    # 计算公平性指数
    response_times = np.array(response_times)
    weighted_times = np.array([response_times[i] * weights[i] for i in range(len(response_times))])
    F_jain = (np.sum(weighted_times) ** 2) / (len(weighted_times) * np.sum(weighted_times ** 2))

    return F_jain, response_times


# 禁忌搜索算法主函数
def tabu_search(n, m_edge, m_cloud, priorities, weights, R_bandwidth, cost_edge, cost_cloud,
                TabuSize, MaxIter, max_cost, T_max, p_user, p_m, r_m, R_edge,
                t_delay_e, t_delay_c, user_data, P_allocation):
    # 初始化一个解
    def initialize_solution():
        server_count = m_edge + m_cloud
        individual = np.zeros((n, server_count), dtype=int)
        server_compute_capability = np.zeros(server_count)
        for i in range(n):
            assigned = False
            while not assigned:
                random_server_idx = np.random.randint(0, server_count)
                if math.ceil((server_compute_capability[random_server_idx] + p_user[i]) / p_m) * r_m <= R_edge[random_server_idx]:
                    individual[i, random_server_idx] = 1
                    server_compute_capability[random_server_idx] += p_user[i]
                    assigned = True
        return individual

    current_solution = initialize_solution()
    best_solution = current_solution.copy()
    best_fitness, best_response_times = compute_fitness(best_solution, n, m_edge, m_cloud, user_data, weights,
                                                        R_bandwidth, t_delay_e, t_delay_c, p_user, P_allocation)
    tabu_list = []
    fitness_history = []

    for _ in tqdm(range(MaxIter), desc="Running Tabu Search"):
        neighbors = generate_neighbors(current_solution, m_edge, m_cloud)
        best_neighbor = None
        best_neighbor_fitness = -1e6
        best_neighbor_response_times = []

        for neighbor in neighbors:
            if not check_constraints(neighbor, n, m_edge, m_cloud, priorities, R_bandwidth,
                                     cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                     user_data, p_user, P_allocation, p_m, r_m, R_edge):
                neighbor = repair_individual(neighbor, n, m_edge, m_cloud, user_data, R_bandwidth, priorities, T_max,
                                             p_user, P_allocation, t_delay_e, t_delay_c, p_m, r_m, R_edge)
            neighbor_fitness, neighbor_response_times = compute_fitness(neighbor, n, m_edge, m_cloud, user_data, weights,
                                                                        R_bandwidth, t_delay_e, t_delay_c, p_user, P_allocation)
            # 修改判断条件
            tabu = any([np.array_equal(neighbor, tabu_neighbor) for tabu_neighbor in tabu_list])
            if (not tabu or neighbor_fitness > best_fitness) and neighbor_fitness > best_neighbor_fitness:
                best_neighbor = neighbor
                best_neighbor_fitness = neighbor_fitness
                best_neighbor_response_times = neighbor_response_times

        if best_neighbor is not None:
            current_solution = best_neighbor
            if best_neighbor_fitness > best_fitness:
                best_solution = best_neighbor
                best_fitness = best_neighbor_fitness
                best_response_times = best_neighbor_response_times
            tabu_list.append(current_solution)
            if len(tabu_list) > TabuSize:
                tabu_list.pop(0)
        fitness_history.append(best_fitness)

    return best_solution, best_fitness, best_response_times, fitness_history