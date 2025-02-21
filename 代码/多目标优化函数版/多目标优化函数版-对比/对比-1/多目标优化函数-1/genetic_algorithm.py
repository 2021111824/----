# 遗传算法模块--genetic_algorithm.py
# 实现遗传算法的核心逻辑，包括种群初始化、适应度计算、交叉、变异等。
import numpy as np
import random
from calculations import compute_response_time, assign_computational_capacity, calculate_response_stats
from cons_new import check_constraints
from tqdm import tqdm
from repair_new import repair_individual


# ========== 遗传算法 ==========
# 1. 初始化种群，同时检查资源约束
def initialize_population(n, server_count, P, user_positions, server_positions, priorities,
                          R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                          cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                          v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net):
    """
    初始化种群，考虑 CPU、内存和带宽资源约束以及响应时间、成本等约束，不满足这些约束的个体重新生成。

    Args:
        n: 用户数
        server_count: 总服务器数
        P: 种群大小
        user_positions: 用户位置
        server_positions: 服务器位置
        priorities: 用户优先级
        R_cpu, R_mem, R_bandwidth: 服务器资源限制
        cpu_demands, mem_demands, bandwidth_demands: 用户资源需求
        cost_edge, cost_cloud: 部署成本参数
        m_edge: 边缘服务器数量
        max_cost: 最大部署预算
        T_max: 最大响应时间约束
        request_sizes: 用户请求大小
        v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net: 计算参数

    Returns:
        population: 初始化后的种群
    """
    population = []
    individual = np.zeros((n, server_count), dtype=int)

    for _ in tqdm(range(P), desc="Initializing Population"):
        valid_individual = False
        attempt_count = 0  # 尝试计数
        while not valid_individual:
            individual = np.zeros((n, server_count), dtype=int)
            server_cpu_usage = np.zeros(server_count)
            server_mem_usage = np.zeros(server_count)
            server_bandwidth_usage = np.zeros(server_count)

            for i in range(n):  # 遍历每个用户
                assigned = False
                while not assigned:  # 为每个用户分配一个服务器，直到满足资源约束
                    random_server_idx = np.random.randint(0, server_count)  # 随机选择一个服务器

                    # 检查该服务器是否资源满足限制
                    if (server_cpu_usage[random_server_idx] + cpu_demands[i] <= R_cpu[random_server_idx] and
                            server_mem_usage[random_server_idx] + mem_demands[i] <= R_mem[random_server_idx] and
                            server_bandwidth_usage[random_server_idx] + bandwidth_demands[i] <= R_bandwidth[
                                random_server_idx]):
                        # 如果满足 CPU、内存和带宽约束，分配用户到服务器
                        individual[i, random_server_idx] = 1
                        server_cpu_usage[random_server_idx] += cpu_demands[i]
                        server_mem_usage[random_server_idx] += mem_demands[i]
                        server_bandwidth_usage[random_server_idx] += bandwidth_demands[i]
                        assigned = True  # 成功分配后退出循环

            # 检查是否满足所有约束
            valid = check_constraints(individual, user_positions, server_positions, priorities,
                                                 R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                                                 cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                                                 v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)
            valid_individual = valid
            attempt_count += 1

            if attempt_count > 100:  # 防止死循环，如果尝试次数过多则跳出
                print("Warning: Too many attempts to generate valid individual. Moving forward.")
                break

        population.append(individual)  # 成功生成的个体加入种群

    return population


# 2. 计算种群中个体的适应度值，基于Jain公平性指数和约束条件
def compute_fitness_new(individual, user_positions, server_positions, request_sizes, priorities,
                        v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge):

    n_users = len(user_positions)

    # 根据分配情况计算每个用户的计算能力
    user_capacities = assign_computational_capacity(individual, user_positions, server_positions, request_sizes, P_edge,
                                                    P_cloud, m_edge, priorities)

    response_times = []

    for i in range(n_users):
        server_idx = np.argmax(individual[i])  # 用户分配到的服务器
        is_edge = server_idx < m_edge  # 是否为边缘服务器

        response_time = compute_response_time(
            user_positions[i], server_positions[server_idx], is_edge, request_sizes[i], user_capacities[i],
            v_edge, v_cloud, b_edge, b_cloud
        )
        response_times.append(response_time)

    # 使用calculate_response_stats计算不同优先级的响应时间统计信息
    response_stats = calculate_response_stats(response_times, priorities)
    sorted_priorities = sorted(response_stats.keys(), reverse=False)  # 按优先级从低到高排序

    # 计算响应时间比例偏差
    total_response_deviation = 0.0

    for i in range(1, len(sorted_priorities)):
        low_priority = i
        high_priority = i + 1

        low_priority_mean = response_stats[low_priority]["mean"]
        high_priority_mean = response_stats[high_priority]["mean"]

        r_i = 1.35  # 获取期望的响应时间比例 r_i

        total_response_deviation += abs((low_priority_mean / high_priority_mean) - r_i)

    # 计算每个优先级的Jain公平性指数，并保存到字典中
    jain_fairness_indices = {}

    for priority in sorted_priorities:
        # 获取当前优先级内所有用户的响应时间
        priority_users_response_times = [response_times[i] for i in range(n_users) if priorities[i] == priority]

        # 计算该优先级内的Jain公平性指数
        jain_fairness_index = compute_jain_fairness(priority_users_response_times)

        # 将Jain公平性指数保存到字典中
        jain_fairness_indices[priority] = jain_fairness_index

    # 使用λ1和λ2加权目标函数
    lambda_1 = 1.0
    lambda_2 = 2.0

    fitness = -(lambda_1 * total_response_deviation +
                lambda_2 * sum((1-jain_fairness_index) for jain_fairness_index in jain_fairness_indices.values()))

    return fitness, total_response_deviation, jain_fairness_indices, response_times


# 计算Jain公平性指数
def compute_jain_fairness(response_times):
    """
    计算给定响应时间列表的Jain公平性指数。
    参数：
        response_times：一个包含多个用户响应时间的列表。
    返回：
        Jain公平性指数。
    """
    n = len(response_times)
    if n == 0:
        return 0  # 防止除零错误

    total_sum = sum(response_times)
    total_square_sum = sum(t ** 2.0 for t in response_times)

    # 计算Jain公平性指数
    F_jain = (total_sum ** 2.0) / (n * total_square_sum)
    return F_jain


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
def mutate(individual, server_positions, m_edge, P_m, priorities):
    """
    对个体进行变异，优先级高的用户变异到边缘服务器。

    :param individual: 当前个体，表示每个用户的服务器分配情况
    :param server_positions: 服务器的位置列表
    :param m_edge: 边缘服务器的数量
    :param P_m: 变异发生的概率
    :param priorities: 每个用户的优先级
    :return: 变异后的个体
    """
    for i in range(len(individual)):
        if random.random() < P_m:  # 以概率P_m发生变异
            # 如果用户的优先级为3（最大优先级），则优先变异到边缘服务器
            if priorities[i] == 3:  # 优先级为3的用户
                # 找到离用户最近的边缘服务器
                edge_server_idx = random.randint(0, m_edge - 1)  # 随机选择一个边缘服务器
                individual[i, edge_server_idx] = 1  # 将用户分配到最近的边缘服务器
            else:
                # 对于其他优先级的用户，仍然进行随机变异
                server_idx = random.randint(0, len(server_positions) - 1)  # 随机选择服务器
                individual[i] = 0  # 重置用户分配
                individual[i, server_idx] = 1  # 将用户分配到随机服务器
    return individual  # 显式返回变异后的个体


# 6. 遗传算法
# 遗传算法主函数，通过不断进化找到最优解决方案。
def genetic_algorithm(user_positions, server_positions, request_sizes, priorities,
                      R_cpu, R_mem, R_bandwidth,
                      cost_edge, cost_cloud,
                      m_edge, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud,
                      P, G_max, P_c, P_m, max_cost,
                      cpu_demands, mem_demands, bandwidth_demands, p_net, T_max):

    """
    遗传算法主函数。
    """
    population = initialize_population(len(user_positions), len(server_positions), P,  user_positions, server_positions, priorities,
                                       R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                                       cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                                       v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)

    # 用来记录最优解
    best_solution = None
    best_fitness = -1e6
    best_jain_fairness = {}
    best_response_deviation = 0.0
    best_response_times = []

    fitness_history = []  # 记录每代最优适应度
    jain_fairness_history = []  # 记录每代对应的Jain公平性指数
    response_deviation_history = []  # 记录每代对应的响应时间比例偏差

    for g in tqdm(range(G_max), desc="Running Genetic Algorithm"):
        fitnesses = []
        response_deviations = []
        jain_fairness_indices = []
        response_times = []

        for ind in population:
            fitness, response_deviation, jain_fairness_indic, response_time = compute_fitness_new(ind, user_positions, server_positions,
                                                                                    request_sizes, priorities,
                                                                                    v_edge, v_cloud, b_edge, b_cloud,
                                                                                    P_edge, P_cloud, m_edge)

            fitnesses.append(fitness)
            response_deviations.append(response_deviation)
            jain_fairness_indices.append(jain_fairness_indic)
            response_times.append(response_time)

        # 记录最优解
        best_idx = np.argmax(fitnesses)  # 获取当前种群中的最优个体索引
        if fitnesses[best_idx] > best_fitness:
            # 更新全局最优
            best_fitness = fitnesses[best_idx]
            best_solution = population[best_idx]
            best_jain_fairness = jain_fairness_indices[best_idx]
            best_response_deviation = response_deviations[best_idx]
            best_response_times = response_times[best_idx]
            # print(f"Generation {g}: Best solution = {best_solution}")
            # print(f"Generation {g}: Best fitness = {best_fitness}")
            # print(f"Generation {g}: Best Jain fairness = {best_jain_fairness}")
            # print(f"Generation {g}: Best response deviation = {best_response_deviation}")
            # print(f"Generation {g}: Best response times = {best_response_times}")

        fitness_history.append(best_fitness)  # 记录每代的最优适应度
        jain_fairness_history.append(best_jain_fairness)  # 记录每代的最优适应度对应的F_jain
        response_deviation_history.append(best_response_deviation)  # 记录每代的最优适应度对应的响应时间偏差

        new_population = [population[best_idx]]  # 保留当前最优个体
        selected_population = select_population(population, fitnesses)  # 选择下一代

        for i in range(1, len(selected_population) - 1, 2):
            if random.random() < P_c:  # 以概率 P_c 进行交叉操作
                # 执行交叉操作
                child1, child2 = crossover(selected_population[i], selected_population[i + 1], len(user_positions))

                # 先检查交叉后的个体是否满足约束
                valid = check_constraints(
                        child1, user_positions, server_positions, priorities,
                        R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                        cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                        v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)
                if not valid:
                    # 如果不满足约束，修复个体
                    child1 = repair_individual(child1, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                                               cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                                               request_sizes, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)

                valid = check_constraints(
                        child2, user_positions, server_positions, priorities,
                        R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                        cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                        v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)
                if not valid:
                    # 如果不满足约束，修复个体
                    child2 = repair_individual(child2, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                                               cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                                               request_sizes, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)

                # 将修复后的子代加入到新种群中
                new_population.extend([child1, child2])
            else:
                # 如果不进行交叉，直接将父代加入新种群
                new_population.extend([selected_population[i], selected_population[i + 1]])
        if len(selected_population) % 2 == 1:  # 如果个体数为奇数，直接添加最后一个个体
            new_population.append(selected_population[-1])

        for idx, ind in enumerate(new_population[1:]):  # 对新种群进行变异
            mutate(ind, server_positions, m_edge, P_m, priorities)

            # 修复不满足约束的个体
            valid = check_constraints(ind, user_positions, server_positions, priorities,
                                     R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                                     cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                                     v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)
            if not valid:
                new_population[idx + 1] = repair_individual(ind, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                                                            cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                                                            request_sizes, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)

        # 遗传算法补充种群逻辑
        while len(new_population) < 10:
            base_individual = random.choice(population)  # 随机选择现有个体
            new_individual = mutate(base_individual.copy(), server_positions, m_edge, P_m, priorities)

            # 显式传递 `request_sizes` 并检查约束
            valid = check_constraints(
                    new_individual, user_positions, server_positions, priorities,
                    R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                    cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                    v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net
            )
            if new_individual is not None and valid:
                new_population.append(new_individual)  # 添加合法个体

        # 更新种群
        population = new_population

    # 返回最优解及其适应度值
    return best_solution, best_fitness, best_jain_fairness, best_response_deviation, best_response_times,\
           fitness_history, jain_fairness_history, response_deviation_history
