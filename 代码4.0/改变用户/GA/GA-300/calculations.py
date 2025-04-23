import numpy as np
import random

# 固定 numpy 和 random 模块的随机种子
np.random.seed(40)
random.seed(40)


# ========== 计算逻辑 ==========
# 新的计算 P_allocation 的函数
def calculate_edge_server_allocation(user_positions, weights, p_user, total_available_computing_resources):
    # 计算所有用户加权计算需求的总和
    total_weighted_p = np.sum(weights * p_user)

    # 初始化 P_allocation
    P_allocation = np.zeros(len(user_positions))

    # 对于每个用户，计算分配的计算能力
    for i in range(len(user_positions)):
        weighted_p_user = weights[i] * p_user[i]  # 用户的加权计算需求
        user_share_of_resources = weighted_p_user / total_weighted_p  # 用户所占资源的比例
        P_allocation[i] = user_share_of_resources * total_available_computing_resources * 0.8  # 计算分配给该用户的计算能力
    print(P_allocation)
    return P_allocation


# 计算加权 Jain 公平性指数
def calculate_weighted_jain_index(individual, n, m_edge, m_cloud, weights, t_delay_e, t_delay_c, user_data,
                                  R_bandwidth, p_user, P_allocation, P_cloud):
    """
    计算加权Jain公平性指数
    """
    user_bandwidth = assign_bandwidth_capacity(individual, n, m_edge, m_cloud, user_data, R_bandwidth)
    server_indices = np.argmax(individual, axis=1)

    # 预计算值，减少重复计算
    t_delay_e_sliced = t_delay_e[np.arange(n), server_indices]
    t_delay_c_sliced = t_delay_c[np.arange(n)]
    p_user_sliced = p_user[np.arange(n)]
    P_allocation_sliced = P_allocation[np.arange(n)]

    weighted_response_times = np.empty(n)
    for i in range(n):
        is_edge = server_indices[i] < m_edge
        response_time = compute_response_time(
            t_delay_e_sliced[i],
            t_delay_c_sliced[i],
            is_edge,
            user_data[i],
            user_bandwidth[i],
            p_user_sliced[i],
            P_allocation_sliced[i],
            P_cloud
        )
        weighted_response_times[i] = response_time * weights[i]

    numerator = np.sum(weighted_response_times) ** 2
    denominator = n * np.sum(weighted_response_times ** 2)
    return numerator / denominator if denominator != 0 else 0


# 根据连接情况进行服务器的带宽分配
def assign_bandwidth_capacity(individual, n, m_edge, m_cloud, user_data, R_bandwidth):
    """
        根据请求大小和优先级分配带宽
    """
    user_bandwidth = np.zeros(n)  # 每个用户分配到的带宽
    assigned_servers = np.argmax(individual, axis=1)

    total_data_demand_edge = np.bincount(assigned_servers[assigned_servers < m_edge],
                                         weights=user_data[assigned_servers < m_edge], minlength=m_edge)
    total_data_demand_cloud = np.bincount(assigned_servers[assigned_servers >= m_edge] - m_edge,
                                          weights=user_data[assigned_servers >= m_edge], minlength=m_cloud)

    for i in range(n):
        user_demand = user_data[i]
        assigned_server = assigned_servers[i]
        if assigned_server < m_edge:
            assigned_bandwidth = R_bandwidth[assigned_server] * (
                user_demand / total_data_demand_edge[assigned_server] if total_data_demand_edge[
                                                                             assigned_server] > 0 else 1)
        else:
            cloud_index = assigned_server - m_edge
            assigned_bandwidth = R_bandwidth[assigned_server] if total_data_demand_cloud[cloud_index] > 0 else 1
        user_bandwidth[i] = assigned_bandwidth * np.random.uniform(0.8, 1.2)

    return user_bandwidth


# 计算响应时间
def compute_response_time(t_delay_e, t_delay_c, is_edge, user_data, user_bandwidth, p_user, P_allocation, P_cloud):
    """
    响应时间计算
    """
    if is_edge:
        # print("edge" + str(p_user / P_allocation))
        return t_delay_e + (user_data / user_bandwidth) * 1000 + (p_user / P_allocation)
    else:
        # print("cloud" + str(p_user / P_cloud))
        return t_delay_c + (user_data / user_bandwidth) * 1000 + (p_user / P_cloud)


# 计算每个用户的响应时间
def compute_response_times(individual, n, m_edge, m_cloud, t_delay_e, t_delay_c, user_data, R_bandwidth, p_user,
                           P_allocation, P_cloud):
    """
    计算每个用户的响应时间
    """
    response_times = np.zeros(n)
    user_bandwidth = assign_bandwidth_capacity(individual, n, m_edge, m_cloud, user_data, R_bandwidth)

    server_indices = np.argmax(individual, axis=1)
    t_delay_e_sliced = t_delay_e[np.arange(n), server_indices]
    t_delay_c_sliced = t_delay_c[np.arange(n)]
    p_user_sliced = p_user[np.arange(n)]
    P_allocation_sliced = P_allocation[np.arange(n)]

    for i in range(n):
        response_times[i] = compute_response_time(t_delay_e_sliced[i], t_delay_c_sliced[i], server_indices[i] < m_edge,
                                                  user_data[i], user_bandwidth[i], p_user_sliced[i],
                                                  P_allocation_sliced[i], P_cloud)

    return response_times


# 统计不同优先级用户的响应时间情况
def calculate_response_stats(response_times, priorities):
    """
    计算不同优先级的响应时间统计信息
    """
    response_times = np.array(response_times)
    stats = {}
    unique_priorities = np.unique(priorities)

    for level in unique_priorities:
        idx = (priorities == level)
        times = response_times[idx]
        stats[level] = {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
        }
    return stats


# 边缘节点成本计算
def calculate_edge_cost(individual, m_edge, cost_edge):
    """
    计算边缘服务器的成本
    """
    edge_cost_details = {"fixed": 0}
    used_edge_servers = np.any(individual[:, :m_edge], axis=0).sum()
    edge_cost_details["fixed"] = used_edge_servers * cost_edge["fixed"]
    total_edge_cost = sum(edge_cost_details.values())
    return total_edge_cost, edge_cost_details


# 云节点成本计算
def calculate_cloud_cost(individual, p_user, cost_cloud, m_edge):
    """
        计算云服务器的成本
    """
    cloud_cost_details = {"p_net": 0}

    cloud_assignments = individual[:, m_edge:]
    cloud_cost_details["p_net"] = np.sum(p_user[:, np.newaxis] * cloud_assignments) * cost_cloud["p_net"]
    total_cloud_cost = sum(cloud_cost_details.values())
    return total_cloud_cost, cloud_cost_details


# 总成本计算
def calculate_total_cost(individual, m_edge, cost_edge, cost_cloud, p_user):
    """
    计算总成本，将边缘和云节点的成本合并
    """
    edge_cost, edge_cost_details = calculate_edge_cost(individual, m_edge, cost_edge)
    cloud_cost, cloud_cost_details = calculate_cloud_cost(individual, p_user, cost_cloud, m_edge)
    total_cost = edge_cost + cloud_cost
    cost_details = {
        "edge": edge_cost_details,
        "cloud": cloud_cost_details,
        "total": total_cost
    }

    return total_cost, cost_details
