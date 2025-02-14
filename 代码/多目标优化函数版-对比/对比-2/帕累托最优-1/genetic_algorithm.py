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
    """
    通过比较目标值来进行非支配排序，确定每个个体的非支配级别。
    Args:
        population: 种群中的个体。
        objectives: 每个个体的目标值。

    Returns: 返回 F，它是一个包含多个前沿层的列表。每个前沿层是一个个体的集合，个体之间的支配关系是按照非支配排序确定的

    """

    S = [[] for _ in range(len(population))]  # S 是一个列表，其中每个元素对应一个个体，S[p] 存储的是个体 p 支配的个体
    n = [0] * len(population)  # n[p] 存储的是个体 p 被支配的个体的数量
    rank = [0] * len(population)  # rank[p] 存储的是个体 p 的非支配排序等级
    F = [[]]  # 二维列表，用于存储不同的非支配前沿。F[i] 存储的是第 i 层的个体

    # 双层循环，判断支配关系
    # 外层循环遍历每个个体 p，内层循环遍历每个其他个体 q
    for p in range(len(population)):
        for q in range(len(population)):
            if p != q:  # 自己不与自己比较
                # 检查个体 p 是否支配个体 q
                # 第一行：p 在所有目标上不比 q 差（Jain指数取了相反数，越小越好；响应时间比例偏差也是越小越好）
                # 第二行：p 至少在一个目标上比 q 更好
                p_dominates_q = all(objectives[p][i] <= objectives[q][i] for i in range(len(objectives[p]))) and \
                                any(objectives[p][i] < objectives[q][i] for i in range(len(objectives[p])))
                # 检查个体 q 是否支配个体 p， 方法同上
                q_dominates_p = all(objectives[q][i] <= objectives[p][i] for i in range(len(objectives[q]))) and \
                                any(objectives[q][i] < objectives[p][i] for i in range(len(objectives[q])))
                # 如果 p 支配 q，则将 q 加入到 S[p] 中
                if p_dominates_q:
                    S[p].append(q)
                #  如果 q 支配 p，则将 n[p] 加 1，表示个体 p 被一个个体支配
                elif q_dominates_p:
                    n[p] += 1
        #  如果 n[p] == 0，意味着个体 p 不被任何个体支配，属于第一个非支配前沿（即Pareto前沿）
        if n[p] == 0:
            rank[p] = 0  # 将其分配给第一个前沿
            F[0].append(p)  # 将其添加到 F[0] 中

    # 遍历并更新每个前沿
    i = 0
    while F[i]:  # F[i] 非空，表示当前还有个体在前沿 i 上，需要处理下一个前沿
        Q = []  # 临时列表，用于存储从前沿 i 推进到下一个前沿的个体
        # F[i] 中的每个个体 p，查看它支配的个体 q
        for p in F[i]:
            for q in S[p]:
                n[q] -= 1  # 对于每个支配的个体 q，减少 n[q]，表示少了一个支配者
                if n[q] == 0:  # n[q] == 0，意味着个体 q 不再被其他个体支配，它属于下一个前沿层
                    rank[q] = i + 1
                    Q.append(q)
        # 结束后，F[i] 中的所有个体都已经处理完
        # 下一轮处理 Q 中的个体，增加前沿层数 i，并将 Q 添加到 F 中
        i += 1
        F.append(Q)
    # 删除空前沿，即最后一个空列表
    # 注：array[-1]代表的是array中的最后一个元素
    del F[-1]
    # 最终返回 F，它是一个包含多个前沿层的列表。
    # F[0] 表示 Pareto 前沿，F[1] 表示第二个前沿，依此类推。
    return F


# 拥挤距离计算
def crowding_distance_assignment(I, objectives):
    """
    根据目标函数的值为个体分配拥挤距离，作为多样性指标。

    Args:
        I: 一个列表，表示当前非支配前沿中的个体的索引集合。这些索引指向当前种群中的个体。
        objectives:二维列表，表示所有个体的目标值。对应于多目标优化中的多个目标函数。

    Returns: 返回一个列表 distances，包含了每个个体的拥挤距离。

    """
    l = len(I)  # 获取当前非支配前沿中个体的数量。
    distances = [0] * l  # 初始化一个与个体数量 l 相同的距离列表，每个个体的拥挤距离初始值为 0。
    num_objectives = len(objectives[0])  # 获取目标的数量（即每个个体有多少个目标）
    # 遍历每一个目标函数
    for m in range(num_objectives):
        # 根据第 m 个目标对索引进行排序
        # range(l)： 生成一个从 0 到 l-1 的数字序列，表示个体的索引。
        # key=lambda x: objectives[I[x]][m]：通过目标函数的值对个体进行排序（目标值从小到大） ，
        # I[x] 是当前个体的索引，objectives[I[x]] 获取该个体的目标值。
        sorted_indices = sorted(range(l), key=lambda x: objectives[I[x]][m])
        # 第一个和最后一个个体的拥挤距离设置为无穷大
        # 边界个体因为前后没有邻居，通常被认为是最广阔的，确保在选择个体时，他们被优先选择（有助于维持种群的多样性）
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        # 获取第一个和最后一个个体的目标值
        f_max = objectives[I[sorted_indices[-1]]][m]  # 当前目标函数的最大值
        f_min = objectives[I[sorted_indices[0]]][m]  # 当前目标函数的最小值
        if f_max - f_min != 0:  # 检查目标值范围是否为0，如果相同（即所有个体在该目标上具有相同的值），则无法计算拥挤距离
            for i in range(1, l - 1):  # 从 第二个个体 到 倒数第二个个体 ，计算它们的拥挤距离
                # 增加了相邻两个个体在目标函数 m 上的差值的标准化值：
                # 前半部分：计算目标函数 m 上相邻两个个体的目标值差值；
                # 后半部分(f_max - f_min)：是目标值的范围，用来归一化差值
                distances[sorted_indices[i]] += (objectives[I[sorted_indices[i + 1]]][m] - objectives[I[sorted_indices[i - 1]]][m]) / (f_max - f_min)
    return distances


# 锦标赛选择
def tournament_selection(population, objectives, tournament_size=2):
    """
    从种群中随机选择 tournament_size 个个体进行比较，选择适应度最好的个体进入下一代
    Args:
        population: 种群
        objectives: 目标函数值
        tournament_size: 锦标赛大小

    Returns:返回选出的个体组成的列表，即锦标赛选择的结果

    """
    selected_indices = []  # 存储每轮锦标赛中选择的个体的索引
    # 对每个个体（即总共进行 len(population) 轮锦标赛）进行循环。每轮锦标赛选择一个个体。
    for _ in range(len(population)):
        # 随机从种群中选择 tournament_size（默认是 2）个个体进行锦标赛。
        tournament = random.sample(range(len(population)), tournament_size)
        best_index = tournament[0]  # 记录当前最优个体的索引。
        for index in tournament[1:]:  # 从锦标赛的第二个个体开始，依次与当前的最优个体进行比较。
            # 如果所有目标函数上的值都不大于当前最优个体的目标值
            if all(objectives[index][i] <= objectives[best_index][i] for i in range(len(objectives[index]))):
                best_index = index  # 更新 best_index 为当前个体
        selected_indices.append(best_index)  # 锦标赛结束后，将当前选出的最优个体的索引添加到selected_indices，表示这轮锦标赛的结果
    return [population[i] for i in selected_indices]


# 1 初始化种群，同时检查资源约束
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
    """

    Args:
        user_positions: 用户的位置列表
        server_positions: 服务器的位置列表
        request_sizes: 每个用户请求的大小
        priorities: 用户的优先级
        R_cpu、R_mem、R_bandwidth: 服务器的 CPU、内存和带宽资源
        cost_edge、cost_cloud: 边缘服务器和云服务器的单位成本
        m_edge、v_edge、v_cloud、b_edge、b_cloud: 边缘和云服务器的其他参数
        P_edge、P_cloud: 边缘和云的服务能力
        P: 种群大小
        G_max: 最大代数（即迭代次数）
        P_c: 交叉概率
        P_m: 变异概率
        max_cost: 最大成本限制
        cpu_demands、mem_demands、bandwidth_demands: 用户的资源需求
        p_net: 网络传输单位成本
        T_max: 最大传输时间限制

    Returns:

    """

    # 生成初始种群
    population = initialize_population(len(user_positions), len(server_positions), P, user_positions,
                                       server_positions, priorities,
                                       R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                                       cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                                       v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net)

    for g in tqdm(range(G_max), desc="Running NSGA-II Algorithm"):
        # 计算目标值
        objectives = []
        # 对每个个体计算目标值，函数返回的是一个包含多个目标的列表
        for ind in population:
            obj = compute_objectives(ind, user_positions, server_positions, request_sizes, priorities,
                                     v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)
            objectives.append(obj)

        # 非支配排序，返回多个支配前沿
        fronts = non_dominated_sort(population, objectives)

        # 选择前沿个体填充新种群
        # 选择当前前沿的个体并按拥挤距离排序，确保每个前沿包含的个体数不超过种群大小P
        new_population = []
        i = 0  # 从第一个前沿开始
        while i < len(fronts) and len(new_population) + len(fronts[i]) <= P:
            distances = crowding_distance_assignment(fronts[i], objectives)
            print(f"Front {i} length: {len(fronts[i])}, Distances length: {len(distances)}")
            # 使用相对索引进行排序
            sorted_front_indices = sorted(range(len(fronts[i])), key=lambda x: distances[x], reverse=True)
            sorted_front = [fronts[i][idx] for idx in sorted_front_indices]
            new_population.extend([population[j] for j in sorted_front])
            i += 1

        # 如果前面选出的个体数量不足，取下一个前沿的个体补充知道种群大小为P
        if i < len(fronts):
            remaining = P - len(new_population)
            distances = crowding_distance_assignment(fronts[i], objectives)
            print(f"Final front length: {len(fronts[i])}, Distances length: {len(distances)}")
            # 使用相对索引进行排序
            sorted_front_indices = sorted(range(len(fronts[i])), key=lambda x: distances[x], reverse=True)
            sorted_front = [fronts[i][idx] for idx in sorted_front_indices]
            new_population.extend([population[j] for j in sorted_front[:remaining]])

        # 生成子代
        # 通过锦标赛选择两个父母个体，以概率 P_c 交叉生成子代；如果不进行交叉，则直接选择父代个体作为子代
        offspring = []
        while len(offspring) < P:  # 持续生成子代，直到子代种群的数量达到预设的种群大小 P
            parents = tournament_selection(new_population, objectives)
            parent1, parent2 = parents[0], parents[1]  # 选择父代
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

                offspring.extend([child1, child2])  # 如果进行了交叉操作，将修复后的子代添加到 offspring 列表中
            else:
                offspring.extend([parent1, parent2])  # 如果未进行交叉操作，则直接将父代添加到 offspring 列表中。

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

        combined_population = new_population + offspring          # 合并父代和子代种群
        # 计算合并后种群的目标函数值
        combined_objectives = []
        for ind in combined_population:
            obj = compute_objectives(ind, user_positions, server_positions, request_sizes, priorities,
                                     v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)
            combined_objectives.append(obj)

        # 对合并后的种群进行非支配排序
        combined_fronts = non_dominated_sort(combined_population, combined_objectives)

        # 选择下一代种群
        # 根据拥挤距离选择出优质个体，组成下一代种群，直到种群大小达到 P
        next_generation = []
        i = 0
        while i < len(combined_fronts) and len(next_generation) + len(combined_fronts[i]) <= P:
            distances = crowding_distance_assignment(combined_fronts[i], combined_objectives)
            # 使用相对索引进行排序（降序），优先选择拥挤距离大的个体，保持种群多样性
            sorted_front_indices = sorted(range(len(combined_fronts[i])), key=lambda x: distances[x], reverse=True)
            sorted_front = [combined_fronts[i][idx] for idx in sorted_front_indices]
            next_generation.extend([combined_population[j] for j in sorted_front])
            i += 1
        # 处理剩余个体
        # 如果添加完某个非支配前沿后，next_generation 列表的长度还未达到 P；
        # 则从下一个非支配前沿中选择部分个体，直到 next_generation 列表的长度达到 P。
        if i < len(combined_fronts):
            remaining = P - len(next_generation)
            distances = crowding_distance_assignment(combined_fronts[i], combined_objectives)
            # 使用相对索引进行排序
            sorted_front_indices = sorted(range(len(combined_fronts[i])), key=lambda x: distances[x], reverse=True)
            sorted_front = [combined_fronts[i][idx] for idx in sorted_front_indices]
            next_generation.extend([combined_population[j] for j in sorted_front[:remaining]])
        # 将 next_generation 列表赋值给 population，作为下一代的种群。
        population = next_generation

    # 最后一代的非支配解作为 Pareto 前沿
    final_objectives = []
    # 遍历最后一代种群 population 中的每个个体
    for ind in population:
        obj = compute_objectives(ind, user_positions, server_positions, request_sizes, priorities,
                                 v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge)
        final_objectives.append(obj)

    final_fronts = non_dominated_sort(population, final_objectives)
    # final_fronts 列表中的第一个非支配前沿对应的个体作为 Pareto 前沿解
    pareto_front = [population[i] for i in final_fronts[0]]

    print("最终的Pareto前沿长度为")
    print(len(pareto_front))
    return pareto_front


