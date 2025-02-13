# 计算模块--calculations.py
# 包含响应时间计算、成本计算等逻辑。
import random

import numpy as np
random.seed(42)
np.random.seed(42)


# ========== 计算逻辑 ==========
def assign_computational_capacity(individual, user_positions, server_positions, request_sizes, P_edge, P_cloud, m_edge,
                                  priorities):
    """
    根据请求大小和优先级分配计算能力
    """
    user_capacities = np.zeros(len(user_positions))  # 每个用户的计算能力

    # 优先级系数
    priority_levels = {1: 1, 2: 2, 3: 3}
    levels = np.array([priority_levels[priority] for priority in priorities])  # 优先级的加权系数

    # 预计算每个服务器的总需求
    total_demand_edge = np.zeros(m_edge)
    total_demand_cloud = np.zeros(len(server_positions) - m_edge)
    for k in range(len(user_positions)):
        for j in range(m_edge):  # 计算边缘服务器的总需求
            if np.argmax(individual[k]) == j:
                total_demand_edge[j] += request_sizes[k] * levels[k]
        for j in range(m_edge, len(server_positions)):  # 计算云服务器的总需求
            if np.argmax(individual[k]) == j:
                total_demand_cloud[j - m_edge] += request_sizes[k] * levels[k]

    for i, user_pos in enumerate(user_positions):
        # 获取加权后的请求大小
        user_demand = request_sizes[i] * levels[i]  # 根据优先级加权的请求大小

        assigned_capacity = 0
        for j in range(m_edge):  # 边缘服务器
            if total_demand_edge[j] > 0:  # 只有当总需求大于0时才进行计算
                assigned_capacity += P_edge * (user_demand / total_demand_edge[j])  # 按该用户在边缘服务器上请求占比分配计算能力
            else:
                assigned_capacity += P_edge  # 如果该服务器没有其他用户，分配所有计算能力给该用户

        for j in range(m_edge, len(server_positions)):  # 云服务器
            if total_demand_cloud[j - m_edge] > 0:  # 只有当总需求大于0时才进行计算
                assigned_capacity += P_cloud * (user_demand / total_demand_cloud[j - m_edge])  # 按该用户在云服务器上请求占比分配计算能力
            else:
                assigned_capacity += P_cloud  # 如果该服务器没有其他用户，分配所有计算能力给该用户

        user_capacities[i] = assigned_capacity

    return user_capacities


def compute_response_time(user, server, is_edge, request_size, user_capacity, v_edge, v_cloud, b_edge, b_cloud):
    """
    响应时间计算，根据是否为边缘服务器分别处理
    """
    d_ij = np.linalg.norm(user - server)  # 计算用户到服务器的距离
    if is_edge:
        return d_ij / v_edge + request_size / b_edge + request_size / user_capacity  # 边缘服务器响应时间
    else:
        return d_ij / v_cloud + request_size / b_cloud + request_size / user_capacity  # 云服务器响应时间


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
def calculate_edge_cost(individual, m_edge, cost_edge, cpu_demands, mem_demands, bandwidth_demands):
    """
    计算边缘服务器的成本
    """
    n_users, n_servers = individual.shape
    edge_cost_details = {"fixed": 0, "cpu": 0, "mem": 0, "bandwidth": 0}

    # 固定成本
    edge_cost_details["fixed"] = m_edge * cost_edge["fixed"]

    # 遍历用户分配情况，累加边缘服务器的资源成本
    for i in range(n_users):  # 遍历所有用户
        for j in range(m_edge):  # 遍历所有边缘服务器
            if individual[i, j] == 1:  # 如果用户分配到边缘服务器
                edge_cost_details["cpu"] += cpu_demands[i] * cost_edge["cpu"]
                edge_cost_details["mem"] += mem_demands[i] * cost_edge["mem"]
                edge_cost_details["bandwidth"] += bandwidth_demands[i] * cost_edge["bandwidth"]

    # 计算总成本
    total_edge_cost = sum(edge_cost_details.values())
    return total_edge_cost, edge_cost_details


# ========== 云节点成本计算 ==========
def calculate_cloud_cost(individual, request_sizes, cpu_demands, mem_demands, bandwidth_demands,
                         cost_cloud, p_net, m_edge):
    """
        计算云服务器的成本
    """
    n_users, n_servers = individual.shape
    assert len(request_sizes) == n_users, f"request_sizes length {len(request_sizes)} does not match n_users {n_users}"
    assert len(cpu_demands) == n_users, f"cpu_demands length {len(cpu_demands)} does not match n_users {n_users}"
    assert len(mem_demands) == n_users, f"mem_demands length {len(mem_demands)} does not match n_users {n_users}"
    assert len(
        bandwidth_demands) == n_users, f"bandwidth_demands length {len(bandwidth_demands)} does not match n_users"
    cloud_cost_details = {"cpu": 0, "mem": 0, "bandwidth": 0, "network": 0}

    for i in range (n_users):  # 遍历所有用户
        for j in range(m_edge, n_servers): # 遍历云服务器
            if individual[i, j] == 1:  # 用户 i 分配到云服务器 j
                cloud_cost_details["cpu"] += cpu_demands[i] * cost_cloud["cpu"]
                cloud_cost_details["mem"] += mem_demands[i] * cost_cloud["mem"]
                cloud_cost_details["bandwidth"] += bandwidth_demands[i] * cost_cloud["bandwidth"]
                cloud_cost_details["network"] += request_sizes[i] * p_net

    # 计算总成本
    total_cloud_cost = sum(cloud_cost_details.values())
    return total_cloud_cost, cloud_cost_details


# ========== 总成本计算 ==========
def calculate_total_cost(individual, m_edge, cost_edge, cost_cloud, cpu_demands, mem_demands, bandwidth_demands, request_sizes, p_net):
    """
    计算总成本，将边缘和云节点的成本合并
    Args:
        individual (ndarray): 用户到服务器的分配矩阵
        m_edge (int): 边缘服务器数量
        cost_edge (dict): 边缘服务器的成本
        cost_cloud (dict): 云服务器的成本
        cpu_demands (list): 用户的 CPU 需求
        mem_demands (list): 用户的内存需求
        bandwidth_demands (list): 用户的带宽需求
        request_sizes (list): 用户的请求大小
        p_net (float): 网络传输成本参数
    Returns:
        tuple: (总成本, 成本细节)
    """
    # 边缘节点成本
    edge_cost, edge_cost_details = calculate_edge_cost(
        individual, m_edge, cost_edge, cpu_demands, mem_demands, bandwidth_demands
    )

    # 云节点成本
    cloud_cost, cloud_cost_details = calculate_cloud_cost(
        individual, request_sizes, cpu_demands, mem_demands, bandwidth_demands,
        cost_cloud, p_net, m_edge
    )

    # 合并总成本
    total_cost = edge_cost + cloud_cost
    cost_details = {
        "edge": edge_cost_details,
        "cloud": cloud_cost_details,
        "total": total_cost
    }

    return total_cost, cost_details
