# 遗传算法模块--genetic_algorithm.py
# 实现遗传算法的核心逻辑，包括种群初始化、适应度计算、交叉、变异等。
import math

import numpy as np
import random
from calculations import compute_response_time, assign_bandwidth_capacity
from constraints import check_constraints
from tqdm import tqdm
from repair import repair_individual


# ========== 遗传算法 ==========
# 1. 初始化种群，同时检查资源约束
def initialize_population(n, m_edge, m_cloud, Population, priorities, R_bandwidth, cost_edge, cost_cloud, max_cost,
                          T_max, p_user, p_m, r_m, R_edge, t_delay_e, t_delay_c, user_data, P_allocation):
    """
    初始化种群，考虑计算资源约束，不满足约束的个体重新生成。

    Returns:
        population: 初始化后的种群
    """
    population = []
    server_count = m_edge + m_cloud
    individual = np.zeros((n, server_count), dtype=int)

    for _ in tqdm(range(Population), desc="Initializing Population"):
        valid_individual = False
        attempt_count = 0  # 尝试计数
        while not valid_individual:
            individual = np.zeros((n, server_count), dtype=int)
            server_compute_capability = np.zeros(server_count)

            for i in range(n):  # 遍历每个用户
                assigned = False
                while not assigned:  # 为每个用户分配一个服务器，直到满足资源约束
                    random_server_idx = np.random.randint(0, server_count)  # 随机选择一个服务器

                    # 检查该服务器是否资源满足限制
                    if math.ceil((server_compute_capability[random_server_idx] + p_user[i]) / p_m) * r_m <= R_edge[random_server_idx]:
                        # 如果满足 CPU、内存和带宽约束，分配用户到服务器
                        individual[i, random_server_idx] = 1
                        server_compute_capability[random_server_idx] += p_user[i]
                        assigned = True  # 成功分配后退出循环

            # 检查是否满足所有约束
            valid_individual = check_constraints(individual, n, m_edge, m_cloud, priorities, R_bandwidth,
                                                 cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                                 user_data, p_user, P_allocation, p_m, r_m, R_edge)

            attempt_count += 1

            if attempt_count > 100:  # 防止死循环，如果尝试次数过多则跳出
                print("Warning: Too many attempts to generate valid individual. Moving forward.")
                break

        population.append(individual)  # 成功生成的个体加入种群

    return population


# 2. 计算种群中个体的适应度值，基于Jain公平性指数和约束条件
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


# 3. 使用锦标赛选择法从当前种群中选择下一代
def select_population(population, fitnesses, tournament_size=3):
    """
    选择下一代种群，基于锦标赛选择法。
    Args:
        population (list): 当前种群
        fitnesses (list): 种群适应度值
        tournament_size (int): 锦标赛大小

    Returns:
        list: 选择后的种群
    """
    tournament_size = min(tournament_size, len(population))  # 动态调整锦标赛大小

    if len(population) < 2:
        raise ValueError("Population size ({len(population)}) is too small to continue the algorithm.")

    selected = []
    for _ in range(len(population)):
        candidates = random.sample(range(len(population)), tournament_size)
        best_candidate = max(candidates, key=lambda x: fitnesses[x])
        selected.append(population[best_candidate])
    return selected


# 4. 交叉操作
# 实现单点交叉操作，生成两个子代。
def crossover(parent1, parent2, n):
    point = random.randint(0, n - 1)  # 随机选择交叉点
    child1 = np.vstack([parent1[:point], parent2[point:]])  # 交叉生成子代
    child2 = np.vstack([parent2[:point], parent1[point:]])
    return child1, child2


# 5. 变异操作
# 对个体进行变异，随机改变用户的服务器分配。
def mutate(individual, m_edge, m_cloud, P_mutation, priorities):
    """
    对个体进行变异，优先级高的用户变异到边缘服务器。
    """
    n_server = m_edge + m_cloud
    for i in range(len(individual)):
        if random.random() < P_mutation:  # 以概率P_m发生变异
            # 如果用户的优先级为3（最大优先级），则优先变异到边缘服务器
            if priorities[i] == 3:  # 优先级为3的用户
                # 找到离用户最近的边缘服务器
                edge_server_idx = random.randint(0, m_edge - 1)  # 随机选择一个边缘服务器
                individual[i] = 0  # 重置用户分配
                individual[i, edge_server_idx] = 1  # 将用户分配到边缘服务器
            else:
                # 对于其他优先级的用户，仍然进行随机变异
                server_idx = random.randint(0, n_server - 1)  # 随机选择服务器
                individual[i] = 0  # 重置用户分配
                individual[i, server_idx] = 1  # 将用户分配到随机服务器
    return individual  # 显式返回变异后的个体


# 6. 遗传算法
# 遗传算法主函数，通过不断进化找到最优解决方案。
def genetic_algorithm(n, m_edge, m_cloud, priorities, weights, R_bandwidth, cost_edge, cost_cloud,
                      Population, G_max, P_crossover, P_mutation, max_cost, T_max, p_user, p_m, r_m, R_edge,
                      t_delay_e, t_delay_c, user_data, P_allocation):

    """
    遗传算法主函数。
    """
    population = initialize_population(n, m_edge, m_cloud, Population, priorities, R_bandwidth, cost_edge, cost_cloud, max_cost,
                                       T_max, p_user, p_m, r_m, R_edge, t_delay_e, t_delay_c, user_data, P_allocation)

    best_solution = None
    best_fitness = -1e6
    best_response_times = []
    fitness_history = []  # 记录每代最优适应度

    for g in tqdm(range(G_max), desc="Running Genetic Algorithm"):
        fitnesses = []
        response_times = []

        for ind in population:
            fitness, response_time = compute_fitness(ind, n, m_edge, m_cloud, user_data, weights,
                                                     R_bandwidth, t_delay_e, t_delay_c, p_user, P_allocation)
            fitnesses.append(fitness)
            response_times.append(response_time)

        best_idx = np.argmax(fitnesses)  # 获取当前种群中的最优个体索引
        if fitnesses[best_idx] > best_fitness:  # 更新全局最优
            best_solution = population[best_idx]
            best_fitness = fitnesses[best_idx]
            best_response_times = response_times[best_idx]

        fitness_history.append(best_fitness)  # 记录每代的最优适应度

        new_population = [population[best_idx]]  # 保留当前最优个体
        selected_population = select_population(population, fitnesses)  # 选择下一代

        for i in range(1, len(selected_population) - 1, 2):
            if random.random() < P_crossover:  # 以概率 P_c 进行交叉操作
                # 执行交叉操作
                child1, child2 = crossover(selected_population[i], selected_population[i + 1], n)

                # 先检查交叉后的个体是否满足约束
                if not check_constraints(child1, n, m_edge, m_cloud, priorities, R_bandwidth,
                                         cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                         user_data, p_user, P_allocation, p_m, r_m, R_edge):
                    # 如果不满足约束，修复个体
                    child1 = repair_individual(child1, n, m_edge, m_cloud, user_data, R_bandwidth, priorities, T_max,
                                               p_user, P_allocation, t_delay_e, t_delay_c, p_m, r_m, R_edge)

                if not check_constraints(child2, n, m_edge, m_cloud, priorities, R_bandwidth,
                                         cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                         user_data, p_user, P_allocation, p_m, r_m, R_edge):
                    # 如果不满足约束，修复个体
                    child2 = repair_individual(child2, n, m_edge, m_cloud, user_data, R_bandwidth, priorities, T_max,
                                               p_user, P_allocation, t_delay_e, t_delay_c, p_m, r_m, R_edge)

                # 将修复后的子代加入到新种群中
                new_population.extend([child1, child2])
            else:
                # 如果不进行交叉，直接将父代加入新种群
                new_population.extend([selected_population[i], selected_population[i + 1]])
        if len(selected_population) % 2 == 1:  # 如果个体数为奇数，直接添加最后一个个体
            new_population.append(selected_population[-1])

        for idx, ind in enumerate(new_population[1:]):  # 对新种群进行变异
            mutate(ind, m_edge, m_cloud, P_mutation, priorities)

            # 修复不满足约束的个体
            if not check_constraints(ind, n, m_edge, m_cloud, priorities, R_bandwidth,
                                     cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                                     user_data, p_user, P_allocation, p_m, r_m, R_edge):
                new_population[idx + 1] = repair_individual(ind, n, m_edge, m_cloud, user_data, R_bandwidth, priorities, T_max,
                                                            p_user, P_allocation, t_delay_e, t_delay_c, p_m, r_m, R_edge)

        # 遗传算法补充种群逻辑
        while len(new_population) < 10:
            base_individual = random.choice(population)  # 随机选择现有个体
            new_individual = mutate(base_individual.copy(), m_edge, m_cloud, P_mutation, priorities)

            # 显式传递 `request_sizes` 并检查约束
            if new_individual is not None and check_constraints(
                    new_individual, n, m_edge, m_cloud, priorities, R_bandwidth,
                    cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                    user_data, p_user, P_allocation, p_m, r_m, R_edge
            ):
                new_population.append(new_individual)  # 添加合法个体

        # 更新种群
        population = new_population

    return best_solution, best_fitness, best_response_times, fitness_history  # 返回最优解及其适应度值。
