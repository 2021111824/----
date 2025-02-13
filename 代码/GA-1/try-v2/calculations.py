# 计算模块--calculations.py
# 包含响应时间计算、成本计算等逻辑。
import numpy as np


# ========== 计算逻辑 ==========
# 计算每个服务器的计算能力
def calculate_server_capacity(R_cpu, R_mem, R_bandwidth):
    cpu_to_capacity_ratio = 10  # 每个 CPU 核心的计算能力 (MB/s)
    memory_to_capacity_ratio = 5  # 每 GB 内存的计算能力 (MB/s)
    bandwidth_to_capacity_ratio = 0.5  # 每 Mbps 带宽的计算能力 (MB/s)

    # 计算计算能力
    P_server = R_cpu * cpu_to_capacity_ratio + R_mem * memory_to_capacity_ratio + R_bandwidth * bandwidth_to_capacity_ratio
    return P_server


# 计算用户在服务器上的计算能力
def calculate_user_capacity(user_idx, server_idx, cpu_demands, mem_demands, bandwidth_demands, R_cpu, R_mem,
                            R_bandwidth):
    # 获取服务器的资源
    total_cpu = R_cpu[server_idx]
    total_mem = R_mem[server_idx]
    total_bandwidth = R_bandwidth[server_idx]

    # 获取用户的资源需求
    user_cpu_demand = cpu_demands[user_idx]
    user_mem_demand = mem_demands[user_idx]
    user_bandwidth_demand = bandwidth_demands[user_idx]

    # 计算用户在服务器上的计算能力
    user_cpu_capacity = (user_cpu_demand / total_cpu) * total_cpu  # 根据服务器的 CPU 资源分配
    user_mem_capacity = (user_mem_demand / total_mem) * total_mem  # 根据服务器的内存资源分配
    user_bandwidth_capacity = (user_bandwidth_demand / total_bandwidth) * total_bandwidth  # 根据带宽分配

    # 计算总的计算能力
    user_total_capacity = user_cpu_capacity + user_mem_capacity + user_bandwidth_capacity
    return user_total_capacity


# 响应时间计算
def compute_response_time(user, server, is_edge, request_size, v_edge, v_cloud, b_edge, b_cloud, individual, cpu_demands, mem_demands, bandwidth_demands, R_cpu, R_mem, R_bandwidth, m_edge, m_cloud,
                          user_positions):
    """
    计算响应时间，根据是否为边缘服务器分别处理
    """
    # 动态计算 P_edge 和 P_cloud 数组
    P_edge = np.zeros(len(individual))  # 每个用户连接到边缘服务器时的计算能力
    P_cloud = np.zeros(len(individual))  # 每个用户连接到云服务器时的计算能力

    # 通过用户位置找到对应的用户索引
    user_idx = -1
    for i, user_position in enumerate(user_positions):
        if np.array_equal(user_position, user):
            user_idx = i
            break
    if user_idx == -1:
        raise ValueError("User position not found in individual.")

    # 遍历每个用户并计算其在相应服务器上的计算能力
    for server_idx in range(m_edge + m_cloud):  # 遍历所有服务器
        if individual[user_idx, server_idx] == 1:  # 如果该用户连接到了该服务器
            # 计算该用户在对应服务器上的计算能力
            user_capacity = calculate_user_capacity(user_idx, server_idx, cpu_demands, mem_demands,
                                                    bandwidth_demands, R_cpu, R_mem, R_bandwidth)

            if server_idx < m_edge:
                # 用户连接到边缘服务器
                P_edge[user_idx] = user_capacity  # 用户在该边缘服务器上分配的计算能力
            else:
                # 用户连接到云服务器
                P_cloud[user_idx] = user_capacity  # 用户在该云服务器上分配的计算能力

    # 计算用户与服务器之间的距离
    d_ij = np.linalg.norm(user - server)

    if is_edge:
        return d_ij / v_edge + request_size / b_edge + request_size / P_edge[user_idx]  # 边缘服务器响应时间
    else:
        return d_ij / v_cloud + request_size / b_cloud + request_size / P_cloud[user_idx]  # 云服务器响应时间


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
