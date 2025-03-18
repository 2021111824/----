# 约束条件检查模块--constraints.py
import numpy as np

from calculations import calculate_total_cost, compute_response_time, calculate_response_stats, assign_computational_capacity


def check_constraints(individual, n, m_edge, m_cloud, priorities,
                      R_compute, R_bandwidth, compute_demands, bandwidth_demands,
                      cost_edge, cost_cloud, max_cost, T_max, request_sizes,
                      P_edge, P_cloud, t_delay_e, t_delay_c):
    """
    检查约束条件是否满足

    Returns:
        bool: 是否满足所有约束条件
    """

    n_users, n_servers = individual.shape

    # 根据分配情况计算每个用户的计算能力
    user_capacities = assign_computational_capacity(individual, n, m_edge, m_cloud, compute_demands, P_edge, P_cloud)

    response_times = []

    # 初始化资源使用统计
    server_compute_usage = np.zeros(n_servers)
    server_bandwidth_usage = np.zeros(n_servers)

    for i in range(n):
        server_idx = np.argmax(individual[i])  # 用户分配到的服务器
        is_edge = server_idx < m_edge  # 是否为边缘服务器

        # 更新资源使用情况
        server_compute_usage[server_idx] += compute_demands[i]
        server_bandwidth_usage[server_idx] += bandwidth_demands[i]

        response_time = compute_response_time(
            t_delay_e, t_delay_c, is_edge, request_sizes[i], user_capacities[i], bandwidth_demands[i])
        response_times.append(response_time)

    # 使用calculate_response_stats计算不同优先级的响应时间统计信息
    response_stats = calculate_response_stats(response_times, priorities)

    # 检查约束1：不同优先级用户的平均响应时间约束
    for priority, stats in response_stats.items():
        avg_response_time = stats["mean"]
        if avg_response_time > T_max[priority]:  # 检查是否超过允许值
            return False

    # 检查约束2：部署成本
    total_cost, _ = calculate_total_cost(
        individual, m_edge, cost_edge, cost_cloud,
        compute_demands, bandwidth_demands, request_sizes
    )
    if total_cost > max_cost:
        return False

    # 检查约束3：边缘节点计算资源限制
    # 检查每个服务器的资源是否超过上限
    for j in range(n_servers):
        if server_compute_usage[j] > R_compute[j]:
            return False
        if server_bandwidth_usage[j] > R_bandwidth[j]:
            return False

    # 检查约束4：用户与服务器的连接
    if not all(individual.sum(axis=1) == 1):  # 每个用户只能分配到一个服务器
        return False

    return True
