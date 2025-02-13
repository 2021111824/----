# 遗传算法模块--genetic_algorithm.py
# 实现遗传算法的核心逻辑，包括种群初始化、适应度计算、交叉、变异等。
import numpy as np
import random
from calculations import compute_response_time
from constraints import check_constraints
from tqdm import tqdm
from repair import repair_individual


# ========== 遗传算法 ==========


# 1. 初始化种群，尽量选择距离最近的，同时检查资源约束
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
            valid_individual = check_constraints(individual, user_positions, server_positions, priorities,
                                                 R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                                                 cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                                                 v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)

            attempt_count += 1

            if attempt_count > 100:  # 防止死循环，如果尝试次数过多则跳出
                print("Warning: Too many attempts to generate valid individual. Moving forward.")
                break

        population.append(individual)  # 成功生成的个体加入种群

    return population

# 2. 计算种群中个体的适应度值，基于最小化响应时间和约束条件
def compute_fitness(individual, user_positions, server_positions, request_sizes, priorities, weights,
                    v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud,
                    R_cpu, R_mem, R_bandwidth,
                    cost_edge, cost_cloud, m_edge, max_cost,
                    cpu_demands, mem_demands, bandwidth_demands, p_net, T_max):
    """
    计算适应度值，目标是最小化总体响应时间。
    """
    response_times = []

    for i in range(len(user_positions)):
        server_idx = np.argmax(individual[i])  # 用户分配到的服务器
        is_edge = server_idx < m_edge  # 是否为边缘服务器

        response_time = compute_response_time(
            user_positions[i], server_positions[server_idx], is_edge, request_sizes[i],
            v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud
        )
        response_times.append(response_time)

    # 调用约束检查函数
    if not check_constraints(
            individual, user_positions, server_positions, priorities,
            R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
            cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
            v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net
    ):
        # 如果不满足约束，尝试进行修复
        individual = repair_individual(individual, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                                       cpu_demands, mem_demands, bandwidth_demands, priorities, T_max, request_sizes,
                                       v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)

        # 重新检查修复后的约束
        if not check_constraints(
                individual, user_positions, server_positions, priorities,
                R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net
        ):
            # 如果修复后仍然不满足约束，可以返回一个低适应度值，或者引入惩罚机制
            # 这里我们使用一个惩罚因子 penalty 来根据约束违反的程度调整适应度
            penalty = calculate_penalty(individual, user_positions, server_positions, request_sizes, priorities,
                                        weights,
                                        R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                                        cost_edge, cost_cloud, m_edge, max_cost, T_max, v_edge, v_cloud,
                                        b_edge, b_cloud, P_edge, P_cloud, p_net)
            return -(np.sum(response_times) + penalty)  # 负值表示适应度值较差

    # 如果满足约束，计算总体响应时间的总和作为适应度
    total_response_time = np.sum(response_times)

    return -total_response_time  # 目标是最小化响应时间


# 计算违反约束的惩罚因子
def calculate_penalty(individual, user_positions, server_positions, request_sizes, priorities, weights,
                      R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                      cost_edge, cost_cloud, m_edge, max_cost, T_max, v_edge, v_cloud,
                      b_edge, b_cloud, P_edge, P_cloud, p_net):
    """
    计算约束违反的惩罚因子，可以根据不同约束的违反情况设置不同的权重。
    """
    penalty = 0

    # 计算约束违反的惩罚（例如：资源约束、成本约束等）
    # 你可以根据不同的约束类型增加权重因子，比如 CPU、内存、带宽、成本等

    # 资源约束违反惩罚（例如：CPU、内存、带宽不足）
    resource_violation_penalty = compute_resource_violation(individual, user_positions, server_positions, R_cpu, R_mem,
                                                            R_bandwidth, cpu_demands, mem_demands, bandwidth_demands)
    penalty += resource_violation_penalty

    # 成本约束违反惩罚
    cost_violation_penalty = compute_cost_violation(individual, cost_edge, cost_cloud, m_edge, max_cost)
    penalty += cost_violation_penalty

    # 添加其他约束的违反惩罚（例如：网络延迟、优先级等）

    return penalty


# 计算资源约束违反的惩罚
def compute_resource_violation(individual, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                               cpu_demands, mem_demands, bandwidth_demands):
    penalty = 0
    # 检查是否超出资源限制（例如，检查每个服务器的CPU、内存、带宽等是否被超过）
    for i in range(len(user_positions)):
        server_idx = np.argmax(individual[i])  # 用户分配到的服务器
        is_edge = server_idx < len(R_cpu)  # 是否为边缘服务器

        if is_edge:
            available_cpu = R_cpu[server_idx]
            available_mem = R_mem[server_idx]
            available_bandwidth = R_bandwidth[server_idx]
        else:
            available_cpu = R_cpu[server_idx]
            available_mem = R_mem[server_idx]
            available_bandwidth = R_bandwidth[server_idx]

        # 如果需求超过可用资源，计算惩罚
        if cpu_demands[i] > available_cpu:
            penalty += 10 * (cpu_demands[i] - available_cpu)
        if mem_demands[i] > available_mem:
            penalty += 5 * (mem_demands[i] - available_mem)
        if bandwidth_demands[i] > available_bandwidth:
            penalty += 2 * (bandwidth_demands[i] - available_bandwidth)

    return penalty


# 计算成本约束违反的惩罚
def compute_cost_violation(individual, cost_edge, cost_cloud, m_edge, max_cost):
    penalty = 0
    # 计算边缘和云服务器的总成本
    total_cost = np.sum(list(cost_edge.values())) + np.sum(list(cost_cloud.values()))  # 获取字典的所有值并求和
    if total_cost > max_cost:
        penalty += (total_cost - max_cost) * 100  # 每超出一个单位的成本，增加惩罚

    return penalty



# 3. 使用轮盘赌选择法从当前种群中选择下一代
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
def mutate(individual, server_count, P_m):
    for i in range(len(individual)):
        if random.random() < P_m:  # 以概率P_m发生变异
            server_idx = random.randint(0, server_count - 1)  # 随机选择服务器
            individual[i] = 0  # 重置用户分配
            individual[i, server_idx] = 1
    return individual  # 显式返回变异后的个体


# 6. 遗传算法
# 遗传算法主函数，通过不断进化找到最优解决方案。
def genetic_algorithm(user_positions, server_positions, request_sizes, priorities, weights,
                      R_cpu, R_mem, R_bandwidth,
                      cost_edge, cost_cloud,
                      m_edge, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud,
                      P, G_max, P_c, P_m, max_cost,
                      cpu_demands, mem_demands, bandwidth_demands, p_net, T_max):

    """
    遗传算法主函数。
    """
    server_count = len(server_positions)
    population = initialize_population(len(user_positions), len(server_positions), P,  user_positions, server_positions, priorities,
                          R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                          cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                          v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)

    best_solution = None
    best_fitness = -1e6
    fitness_history = []  # 记录每代最优适应度

    for g in tqdm(range(G_max), desc="Running Genetic Algorithm"):
        fitnesses = [compute_fitness(ind, user_positions, server_positions, request_sizes, priorities, weights,
                                     v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud,
                                     R_cpu, R_mem, R_bandwidth,
                                     cost_edge, cost_cloud, m_edge, max_cost,
                                     cpu_demands, mem_demands, bandwidth_demands, p_net, T_max)
                     for ind in population]

        best_idx = np.argmax(fitnesses)  # 获取当前种群中的最优个体索引
        if fitnesses[best_idx] > best_fitness:  # 更新全局最优
            best_solution = population[best_idx]
            best_fitness = fitnesses[best_idx]

        fitness_history.append(best_fitness)  # 记录每代的最优适应度
        new_population = [population[best_idx]]  # 保留当前最优个体
        selected_population = select_population(population, fitnesses)  # 选择下一代

        for i in range(1, len(selected_population) - 1, 2):
            if random.random() < P_c:  # 以概率 P_c 进行交叉操作
                # 执行交叉操作
                child1, child2 = crossover(selected_population[i], selected_population[i + 1], len(user_positions))

                # 先检查交叉后的个体是否满足约束
                if not check_constraints(
                        child1, user_positions, server_positions, priorities,
                        R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                        cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                        v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net
                ):
                    # 如果不满足约束，修复个体
                    child1 = repair_individual(child1, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                                               cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                                               request_sizes, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)

                if not check_constraints(
                        child2, user_positions, server_positions, priorities,
                        R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                        cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                        v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net
                ):
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
            mutate(ind, server_count, P_m)

            # 修复不满足约束的个体
            if not check_constraints(ind, user_positions, server_positions, priorities,
                                     R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                                     cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                                     v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net):
                new_population[idx + 1] = repair_individual(ind, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                                                            cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                                                            request_sizes, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)

        # 遗传算法补充种群逻辑
        while len(new_population) < 10:
            base_individual = random.choice(population)  # 随机选择现有个体
            new_individual = mutate(base_individual.copy(), len(server_positions), P_m)

            # 显式传递 `request_sizes` 并检查约束
            if new_individual is not None and check_constraints(
                    new_individual, user_positions, server_positions, priorities,
                    R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                    cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                    v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net
            ):
                new_population.append(new_individual)  # 添加合法个体

        # 更新种群
        population = new_population

    return best_solution, best_fitness, fitness_history  # 返回最优解及其适应度值。
