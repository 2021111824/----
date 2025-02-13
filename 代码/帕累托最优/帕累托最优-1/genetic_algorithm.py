import numpy as np
import random
from calculations import compute_response_time, assign_computational_capacity, calculate_response_stats
from cons_new import check_constraints
from tqdm import tqdm
from repair_new import repair_individual


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


# 计算每个个体的两个目标值：Jain公平性指数和响应时间偏差
def compute_objectives(individual, user_positions, server_positions, request_sizes, priorities,
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

    # 我们希望最大化Jain公平性，最小化响应时间偏差
    jain_score = sum(jain_fairness_indices.values())
    return -jain_score, total_response_deviation


# 非支配排序
def non_dominated_sort(population, objectives):
    S = [[] for _ in range(len(population))]
    n = [0] * len(population)
    rank = [0] * len(population)
    F = [[]]
    for p in range(len(population)):
        for q in range(len(population)):
            if p != q:
                p_dominates_q = all(objectives[p][i] <= objectives[q][i] for i in range(len(objectives[p]))) and \
                                any(objectives[p][i] < objectives[q][i] for i in range(len(objectives[p])))
                q_dominates_p = all(objectives[q][i] <= objectives[p][i] for i in range(len(objectives[q]))) and \
                                any(objectives[q][i] < objectives[p][i] for i in range(len(objectives[q])))
                if p_dominates_q:
                    S[p].append(q)
                elif q_dominates_p:
                    n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            F[0].append(p)
    i = 0
    while F[i]:
        Q = []
        for p in F[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        F.append(Q)
    del F[-1]
    return F


# 拥挤距离计算
def crowding_distance_assignment(I, objectives):
    l = len(I)
    distances = [0] * l
    num_objectives = len(objectives[0])
    for m in range(num_objectives):
        # 根据第 m 个目标对索引进行排序
        sorted_indices = sorted(range(l), key=lambda x: objectives[I[x]][m])
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        f_max = objectives[I[sorted_indices[-1]]][m]
        f_min = objectives[I[sorted_indices[0]]][m]
        if f_max - f_min != 0:
            for i in range(1, l - 1):
                distances[sorted_indices[i]] += (objectives[I[sorted_indices[i + 1]]][m] - objectives[I[sorted_indices[i - 1]]][m]) / (f_max - f_min)
    return distances


# 锦标赛选择
def tournament_selection(population, objectives, tournament_size=2):
    selected_indices = []
    for _ in range(len(population)):
        tournament = random.sample(range(len(population)), tournament_size)
        best_index = tournament[0]
        for index in tournament[1:]:
            if all(objectives[index][i] <= objectives[best_index][i] for i in range(len(objectives[index]))):
                best_index = index
        selected_indices.append(best_index)
    return [population[i] for i in selected_indices]


# 1.1 初始化种群，同时检查资源约束
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


# NSGA - II主函数
def nsga_ii(user_positions, server_positions, request_sizes, priorities,
            R_cpu, R_mem, R_bandwidth,
            cost_edge, cost_cloud,
            m_edge, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud,
            P, G_max, P_c, P_m, max_cost,
            cpu_demands, mem_demands, bandwidth_demands, p_net, T_max):
    population = initialize_population(len(user_positions), len(server_positions), P, user_positions,
                                       server_positions, priorities,
                                       R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                                       cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                                       v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)

    for g in tqdm(range(G_max), desc="Running NSGA-II Algorithm"):
        # 计算目标值
        objectives = []
        for ind in population:
            obj = compute_objectives(ind, user_positions, server_positions, request_sizes, priorities,
                                     v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)
            objectives.append(obj)

        # 非支配排序
        fronts = non_dominated_sort(population, objectives)

        new_population = []
        i = 0
        while i < len(fronts) and len(new_population) + len(fronts[i]) <= P:
            distances = crowding_distance_assignment(fronts[i], objectives)
            print(f"Front {i} length: {len(fronts[i])}, Distances length: {len(distances)}")
            # 使用相对索引进行排序
            sorted_front_indices = sorted(range(len(fronts[i])), key=lambda x: distances[x], reverse=True)
            sorted_front = [fronts[i][idx] for idx in sorted_front_indices]
            new_population.extend([population[j] for j in sorted_front])
            i += 1

        if i < len(fronts):
            remaining = P - len(new_population)
            distances = crowding_distance_assignment(fronts[i], objectives)
            print(f"Final front length: {len(fronts[i])}, Distances length: {len(distances)}")
            # 使用相对索引进行排序
            sorted_front_indices = sorted(range(len(fronts[i])), key=lambda x: distances[x], reverse=True)
            sorted_front = [fronts[i][idx] for idx in sorted_front_indices]
            new_population.extend([population[j] for j in sorted_front[:remaining]])

        offspring = []
        while len(offspring) < P:
            parents = tournament_selection(new_population, objectives)
            parent1, parent2 = parents[0], parents[1]
            if random.random() < P_c:  # 以概率 P_c 进行交叉操作
                child1, child2 = crossover(parent1, parent2, len(user_positions))

                # 检查交叉后的个体是否满足约束
                valid1 = check_constraints(
                    child1, user_positions, server_positions, priorities,
                    R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                    cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                    v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)
                if not valid1:
                    child1 = repair_individual(child1, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                                               cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                                               request_sizes, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)

                valid2 = check_constraints(
                    child2, user_positions, server_positions, priorities,
                    R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                    cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                    v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)
                if not valid2:
                    child2 = repair_individual(child2, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                                               cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                                               request_sizes, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)

                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])

        # 对新生成的子代进行变异
        for idx, ind in enumerate(offspring):
            mutated_ind = mutate(ind, server_positions, m_edge, P_m, priorities)

            # 检查变异后的个体是否满足约束
            valid = check_constraints(
                mutated_ind, user_positions, server_positions, priorities,
                R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)
            if not valid:
                offspring[idx] = repair_individual(mutated_ind, user_positions, server_positions, R_cpu, R_mem,
                                                   R_bandwidth,
                                                   cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                                                   request_sizes, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud,
                                                   m_edge)
            else:
                offspring[idx] = mutated_ind

        # 合并父代和子代种群
        combined_population = new_population + offspring
        combined_objectives = []
        for ind in combined_population:
            obj = compute_objectives(ind, user_positions, server_positions, request_sizes, priorities,
                                     v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)
            combined_objectives.append(obj)

        # 对合并后的种群进行非支配排序
        combined_fronts = non_dominated_sort(combined_population, combined_objectives)

        next_generation = []
        i = 0
        while i < len(combined_fronts) and len(next_generation) + len(combined_fronts[i]) <= P:
            distances = crowding_distance_assignment(combined_fronts[i], combined_objectives)
            # 使用相对索引进行排序
            sorted_front_indices = sorted(range(len(combined_fronts[i])), key=lambda x: distances[x], reverse=True)
            sorted_front = [combined_fronts[i][idx] for idx in sorted_front_indices]
            next_generation.extend([combined_population[j] for j in sorted_front])
            i += 1

        if i < len(combined_fronts):
            remaining = P - len(next_generation)
            distances = crowding_distance_assignment(combined_fronts[i], combined_objectives)
            # 使用相对索引进行排序
            sorted_front_indices = sorted(range(len(combined_fronts[i])), key=lambda x: distances[x], reverse=True)
            sorted_front = [combined_fronts[i][idx] for idx in sorted_front_indices]
            next_generation.extend([combined_population[j] for j in sorted_front[:remaining]])

        population = next_generation

    # 最后一代的非支配解作为 Pareto 前沿
    final_objectives = []
    for ind in population:
        obj = compute_objectives(ind, user_positions, server_positions, request_sizes, priorities,
                                 v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)
        final_objectives.append(obj)

    final_fronts = non_dominated_sort(population, final_objectives)
    pareto_front = [population[i] for i in final_fronts[0]]

    return pareto_front


