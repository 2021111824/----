# 计算模块--calculations.py
# 包含响应时间计算、成本计算等逻辑。
import numpy as np


# ========== 计算逻辑 ==========
def assign_computational_capacity(individual, n, m_edge, m_cloud, compute_demands, P_edge, P_cloud):
    """
    根据请求大小和优先级分配计算能力
    """
    user_capacities = np.zeros(n)  # 每个用户的计算能力

    # 预计算每个服务器的计算资源总需求
    total_compute_demand_edge = np.zeros(m_edge)
    total_compute_demand_cloud = np.zeros(m_cloud)

    for k in range(n):
        for j in range(m_edge):  # 计算边缘服务器的总计算资源需求
            if np.argmax(individual[k]) == j:
                total_compute_demand_edge[j] += compute_demands[k]
        for j in range(m_edge, m_edge + m_cloud):  # 计算云服务器的总计算资源需求
            if np.argmax(individual[k]) == j:
                total_compute_demand_cloud[j - m_edge] += compute_demands[k]

    for i in range(n):
        # 计算每个用户获得的计算能力
        user_demand = compute_demands[i]  # 用户的计算资源需求大小

        assigned_capacity = 0
        for j in range(m_edge):  # 边缘服务器
            if total_compute_demand_edge[j] > 0:  # 只有当总需求大于0时才进行计算
                assigned_capacity += P_edge * (user_demand / total_compute_demand_edge[j])  # 按该用户在边缘服务器上请求占比分配计算能力
            else:
                assigned_capacity += P_edge  # 如果该服务器没有其他用户，分配所有计算能力给该用户

        for j in range(m_edge, m_edge + m_cloud):  # 云服务器
            if total_compute_demand_cloud[j - m_edge] > 0:  # 只有当总需求大于0时才进行计算
                assigned_capacity += P_cloud * (user_demand / total_compute_demand_cloud[j - m_edge])
            else:
                assigned_capacity += P_cloud

        user_capacities[i] = assigned_capacity

    return user_capacities


def compute_response_time(t_delay_e, t_delay_c, is_edge, request_size, user_capacity, bandwidth_demand):
    """
    响应时间计算，根据是否为边缘服务器分别处理
    总时延（ms） = 传播时延（延迟） + 传输实验（总数据量/带宽） + 处理时延（总数据量/计算能力）
    """
    if is_edge:
        return (t_delay_e + (request_size / bandwidth_demand) + (request_size / user_capacity)) * 1000  # 边缘服务器响应时间
    else:
        return (t_delay_c + (request_size / bandwidth_demand) + (request_size / user_capacity)) * 1000  # 云服务器响应时间


def calculate_response_stats(response_times, priorities):
    """
    计算不同优先级的响应时间统计信息

    Args:
        response_times (list): 所有用户的响应时间列表
        priorities (list): 所有用户的优先级列表

    Returns:
        dict: 每个优先级的统计信息，包括均值、标准差、最小值和最大值
    """
    response_times = np.array(response_times)  # 确保响应时间是 NumPy 数组
    stats = {}
    for level in np.unique(priorities):  # 遍历每种优先级
        idx = np.where(priorities == level)[0]  # 找到对应优先级的用户索引
        times = response_times[idx]  # 提取该优先级的响应时间
        stats[level] = {
            "mean": np.mean(times),  # 平均响应时间
            "std": np.std(times),  # 响应时间标准差
            "min": np.min(times),  # 最小响应时间
            "max": np.max(times),  # 最大响应时间
        }
    return stats


# ========== 边缘节点成本计算 ==========
def calculate_edge_cost(individual, m_edge, cost_edge, compute_demands, bandwidth_demands):
    """
    计算边缘服务器的成本
    """
    n_users, n_servers = individual.shape

    edge_cost_details = {"fixed": 0, "compute": 0, "bandwidth": 0}

    # 固定成本
    edge_cost_details["fixed"] = m_edge * cost_edge["fixed"]

    # 遍历用户分配情况，累加边缘服务器的资源成本
    for i in range(n_users):  # 遍历所有用户
        for j in range(m_edge):  # 遍历所有边缘服务器
            if individual[i, j] == 1:  # 如果用户分配到边缘服务器
                edge_cost_details["compute"] += compute_demands[i] * cost_edge["compute"]
                edge_cost_details["bandwidth"] += bandwidth_demands[i] * cost_edge["bandwidth"]

    # 计算总成本
    total_edge_cost = sum(edge_cost_details.values())
    return total_edge_cost, edge_cost_details


# ========== 云节点成本计算 ==========
def calculate_cloud_cost(individual, request_sizes, compute_demands, bandwidth_demands, cost_cloud, m_edge):
    """
        计算云服务器的成本
    """
    n_users, n_servers = individual.shape

    cloud_cost_details = {"p_net": 0, "compute": 0, "bandwidth": 0}

    for i in range(n_users):  # 遍历所有用户
        for j in range(m_edge, n_servers):  # 遍历云服务器
            if individual[i, j] == 1:  # 用户 i 分配到云服务器 j
                cloud_cost_details["compute"] += compute_demands[i] * cost_cloud["compute"]
                cloud_cost_details["bandwidth"] += bandwidth_demands[i] * cost_cloud["bandwidth"]
                cloud_cost_details["p_net"] += request_sizes[i] * cost_cloud["p_net"]

    # 计算总成本
    total_cloud_cost = sum(cloud_cost_details.values())
    return total_cloud_cost, cloud_cost_details


# ========== 总成本计算 ==========
def calculate_total_cost(individual, m_edge, cost_edge, cost_cloud, compute_demands, bandwidth_demands, request_sizes):
    """
    计算总成本，将边缘和云节点的成本合并
    Args:
        individual (ndarray): 用户到服务器的分配矩阵
        m_edge (int): 边缘服务器数量
        cost_edge (dict): 边缘服务器的成本
        cost_cloud (dict): 云服务器的成本
        compute_demands (list): 用户的 计算资源 需求
        bandwidth_demands (list): 用户的带宽需求
        request_sizes (list): 用户的请求大小
    Returns:
        tuple: (总成本, 成本细节)
    """
    # 边缘节点成本
    edge_cost, edge_cost_details = calculate_edge_cost(
        individual, m_edge, cost_edge, compute_demands, bandwidth_demands
    )

    # 云节点成本
    cloud_cost, cloud_cost_details = calculate_cloud_cost(
        individual, request_sizes, compute_demands, bandwidth_demands, cost_cloud, m_edge
    )

    # 合并总成本
    total_cost = edge_cost + cloud_cost
    cost_details = {
        "edge": edge_cost_details,
        "cloud": cloud_cost_details,
        "total": total_cost
    }

    return total_cost, cost_details
