# 计算模块--calculations.py
# 包含响应时间计算、成本计算等逻辑。
import numpy as np
import random
# 固定 numpy 的随机种子
np.random.seed(48)
# 固定 random 模块的随机种子
random.seed(48)


# ========== 计算逻辑 ==========
# 根据连接情况进行服务器的带宽分配
def assign_bandwidth_capacity(individual, n, m_edge, m_cloud, user_data, R_bandwidth):
    """
        根据请求大小和优先级分配带宽
    """
    user_bandwidth = np.zeros(n)  # 每个用户分配到的带宽

    # 预计算每个服务器的总数据量
    total_data_demand_edge = np.zeros(m_edge)
    total_data_demand_cloud = np.zeros(m_cloud)

    for k in range(n):
        for j in range(m_edge):  # 计算边缘服务器的总数据量
            if np.argmax(individual[k]) == j:
                total_data_demand_edge[j] += user_data[k]
        for j in range(m_edge, m_edge + m_cloud):  # 计算云服务器的总计算资源需求
            if np.argmax(individual[k]) == j:
                total_data_demand_cloud[j - m_edge] += user_data[k]

    for i in range(n):
        # 计算每个用户获得的带宽
        user_demand = user_data[i]  # 用户的计算资源需求大小

        assigned_bandwidth = 0
        for j in range(m_edge):  # 边缘服务器
            if np.argmax(individual[i]) == j:
                if total_data_demand_edge[j] > 0:  # 只有当总需求大于0时才进行计算
                    assigned_bandwidth = R_bandwidth[j] * (user_demand / total_data_demand_edge[j])  # 按该用户在边缘服务器上请求占比分配带宽
                else:
                    assigned_bandwidth = R_bandwidth[j]  # 如果该服务器没有其他用户，分配所有带宽给该用户

        for j in range(m_edge, m_edge + m_cloud):  # 云服务器
            if np.argmax(individual[i]) == j:
                if total_data_demand_cloud[j - m_edge] > 0:  # 只有当总需求大于0时才进行计算
                    assigned_bandwidth = R_bandwidth[j]

        user_bandwidth[i] = assigned_bandwidth * np.random.uniform(0.8, 1.2)

    return user_bandwidth


# 计算各个用户的响应时间
def compute_response_time(t_delay_e, t_delay_c, is_edge, user_data, user_bandwidth, p_user, P_allocation):
    """
    响应时间计算，根据是否为边缘服务器分别处理
    总时延（ms） = 传播时延（延迟） + 传输实验（总数据量/带宽） + 处理时延（总计算需求/计算能力）
    """
    if is_edge:
        # print("连接到边缘服务器")
        # print("用户数据量：" + str(user_data) + "Mbit；用户带宽：" + str(user_bandwidth) + "Mbps；用户计算单位需求：" + str(p_user) + "：分配给用户的计算能力：" + str(P_allocation))
        # print("传播延迟：" + str(t_delay_e) + "ms；传输延迟：" + str((user_data / user_bandwidth) * 1000) + "ms；处理延迟：" + str((p_user / P_allocation)) + "ms")
        # print("总时延：" + str(t_delay_e + (user_data / user_bandwidth) * 1000 + (p_user / P_allocation)) + "ms")
        return t_delay_e + (user_data / user_bandwidth) * 1000 + (p_user / P_allocation)  # 边缘服务器响应时间
    else:
        # print("连接到云服务器")
        # print("用户数据量：" + str(user_data) + "Mbit；用户带宽：" + str(user_bandwidth) + "Mbps；用户计算单位需求：" + str(
        #     p_user) + "：分配给用户的计算能力：" + str((P_allocation * 10)))
        # print("传播延迟：" + str(t_delay_c) + "ms；传输延迟：" + str((user_data / user_bandwidth) * 1000) + "ms；处理延迟：" + str(p_user / (P_allocation * 10)) + "ms")
        # print("总时延：" + str(t_delay_c + (user_data / user_bandwidth) * 1000 + (p_user / (P_allocation * 10))) + "ms")
        return t_delay_c + (user_data / user_bandwidth) * 1000 + (p_user / (P_allocation * 10))  # 云服务器响应时间


# 统计不同优先级用户的响应时间情况
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


# 边缘节点成本计算
def calculate_edge_cost(individual, m_edge, cost_edge):
    """
    计算边缘服务器的成本
    """

    edge_cost_details = {"fixed": 0}

    # 计算真正被使用的边缘服务器数量
    used_edge_servers = np.any(individual[:, :m_edge], axis=0).sum()

    # 计算固定成本（只计算使用的服务器数量）
    edge_cost_details["fixed"] = used_edge_servers * cost_edge["fixed"]

    # 计算总成本
    total_edge_cost = sum(edge_cost_details.values())
    return total_edge_cost, edge_cost_details


# 云节点成本计算
def calculate_cloud_cost(individual, p_user, cost_cloud, m_edge):
    """
        计算云服务器的成本
    """
    n_users, n_servers = individual.shape

    cloud_cost_details = {"p_net": 0}

    for i in range(n_users):  # 遍历所有用户
        for j in range(m_edge, n_servers):  # 遍历云服务器
            if individual[i, j] == 1:  # 用户 i 分配到云服务器 j
                cloud_cost_details["p_net"] += p_user[i] * cost_cloud["p_net"]

    # 计算总成本
    total_cloud_cost = sum(cloud_cost_details.values())
    return total_cloud_cost, cloud_cost_details


# 总成本计算
def calculate_total_cost(individual, m_edge, cost_edge, cost_cloud, p_user):
    """
    计算总成本，将边缘和云节点的成本合并
    """
    # 边缘节点成本
    edge_cost, edge_cost_details = calculate_edge_cost(
        individual, m_edge, cost_edge
    )

    # 云节点成本
    cloud_cost, cloud_cost_details = calculate_cloud_cost(
        individual, p_user, cost_cloud, m_edge
    )

    # 合并总成本
    total_cost = edge_cost + cloud_cost
    cost_details = {
        "edge": edge_cost_details,
        "cloud": cloud_cost_details,
        "total": total_cost
    }

    return total_cost, cost_details
