import numpy as np
from calculations import assign_bandwidth_capacity, calculate_total_cost, compute_response_time, calculate_response_stats


def check_constraints(individual, n, m_edge, m_cloud, priorities, R_bandwidth,
                      cost_edge, cost_cloud, max_cost, T_max, t_delay_e, t_delay_c,
                      user_data, p_user, P_allocation, p_m, r_m, R_edge, P_cloud):
    """
    检查约束条件是否满足

    检查 1：不同优先级用户的平均响应时间约束
    检查 2：成本约束
    棌查 3：边缘服务器计算资源约束
    检查 4：用户与服务器的连接约束

    Returns:
        bool: 是否满足所有约束条件
    """

    n_users, n_servers = individual.shape

    # 计算每个用户的带宽
    user_bandwidth = assign_bandwidth_capacity(individual, n_users, m_edge, m_cloud, user_data, R_bandwidth)

    # 初始化响应时间列表和服务器计算能力需求
    response_times = np.zeros(n_users)
    server_compute_capability = np.zeros(n_servers)

    # 计算每个用户的响应时间以及更新服务器计算能力
    for i in range(n_users):
        server_idx = np.argmax(individual[i])  # 用户分配的服务器
        is_edge = server_idx < m_edge  # 判断是否为边缘服务器

        # 更新服务器计算能力需求
        server_compute_capability[server_idx] += P_allocation[i]

        # 计算用户的响应时间
        response_times[i] = compute_response_time(
            t_delay_e[i][server_idx], t_delay_c[i], is_edge, user_data[i], user_bandwidth[i], p_user[i], P_allocation[i], P_cloud
        )

    # 计算每个服务器的计算资源使用
    server_compute_resource_usage = (np.ceil(server_compute_capability / p_m)) * r_m

    # 使用 calculate_response_stats 计算不同优先级的响应时间统计信息
    response_stats = calculate_response_stats(response_times, priorities)

    # 检查约束1：不同优先级用户的平均响应时间
    for priority, stats in response_stats.items():
        avg_response_time = stats["mean"]
        if avg_response_time > T_max[priority]:  # 超过最大响应时间限制
            return False

    # 检查约束2：总成本
    total_cost, _ = calculate_total_cost(individual, m_edge, cost_edge, cost_cloud, p_user)
    if total_cost > max_cost:  # 超过最大成本
        return False

    # 检查约束3：边缘服务器计算资源约束
    if np.any(server_compute_resource_usage > R_edge):  # 检查是否超过边缘服务器资源
        return False

    # 检查约束4：每个用户只能分配到一个服务器
    if not np.all(individual.sum(axis=1) == 1):  # 每个用户分配一个服务器
        return False

    return True
