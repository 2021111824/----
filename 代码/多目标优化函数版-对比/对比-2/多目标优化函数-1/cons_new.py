# 约束条件检查模块--constraints.py
import numpy as np

from calculations import calculate_total_cost, compute_response_time, calculate_response_stats, assign_computational_capacity


def check_constraints(individual, user_positions, server_positions, priorities,
                      R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands,
                      cost_edge, cost_cloud, m_edge, max_cost, T_max, request_sizes,
                      v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, p_net):
    """
    检查约束条件是否满足

    Args:
        individual (ndarray): 用户到服务器的分配矩阵
        user_positions (ndarray): 用户位置
        server_positions (ndarray): 服务器位置
        priorities (ndarray): 用户优先级
        R_cpu, R_mem, R_bandwidth (ndarray): 服务器资源限制
        cpu_demands, mem_demands, bandwidth_demands (ndarray): 用户资源需求
        cost_edge, cost_cloud (dict): 部署成本参数
        m_edge (int): 边缘服务器数量
        max_cost (float): 最大部署预算
        T_max (dict): 每个优先级的最大平均响应时间约束，格式为{priority: max_time}
        request_sizes (ndarray): 用户请求大小

    Returns:
        bool: 是否满足所有约束条件
    """

    n_users, n_servers = individual.shape

    # 根据分配情况计算每个用户的计算能力
    user_capacities = assign_computational_capacity(individual, user_positions, server_positions, request_sizes, P_edge, P_cloud,
                                                    m_edge, priorities)

    response_times = []

    # 初始化资源使用统计
    server_cpu_usage = np.zeros(n_servers)
    server_mem_usage = np.zeros(n_servers)
    server_bandwidth_usage = np.zeros(n_servers)

    for i in range(n_users):
        server_idx = np.argmax(individual[i])  # 用户分配到的服务器
        is_edge = server_idx < m_edge  # 是否为边缘服务器

        # 更新资源使用情况
        server_cpu_usage[server_idx] += cpu_demands[i]
        server_mem_usage[server_idx] += mem_demands[i]
        server_bandwidth_usage[server_idx] += bandwidth_demands[i]

        response_time = compute_response_time(
            user_positions[i], server_positions[server_idx], is_edge, request_sizes[i], user_capacities[i],
            v_edge, v_cloud, b_edge, b_cloud
        )
        response_times.append(response_time)

    # 使用calculate_response_stats计算不同优先级的响应时间统计信息
    response_stats = calculate_response_stats(response_times, priorities)
    valid = True  # 默认所有约束满足

    # 检查约束1：不同优先级用户的平均响应时间约束
    for priority, stats in response_stats.items():
        avg_response_time = stats["mean"]
        if avg_response_time > T_max[priority]:  # 检查是否超过允许值
            valid = False
            break  # 一旦违反就终止

    # 检查约束2：部署成本
    total_cost, _ = calculate_total_cost(
        individual, m_edge, cost_edge, cost_cloud,
        cpu_demands, mem_demands, bandwidth_demands, request_sizes, p_net
    )
    if total_cost > max_cost:
        valid = False

    # 检查约束3：边缘节点计算资源限制
    # 检查每个服务器的资源是否超过上限
    for j in range(n_servers):
        if server_cpu_usage[j] > R_cpu[j] or server_mem_usage[j] > R_mem[j] or server_bandwidth_usage[j] > R_bandwidth[j]:
            valid = False  # 严格约束失败
            break

    # 检查约束4：用户与服务器的连接
    if not all(individual.sum(axis=1) == 1):  # 每个用户只能分配到一个服务器
        valid = False  # 严格约束失败

    return valid
# , user_capacities, response_times, response_stats, server_cpu_usage, server_mem_usage, server_bandwidth_usage
