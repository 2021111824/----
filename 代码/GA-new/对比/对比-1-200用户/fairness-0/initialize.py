# 初始化模块--initialize.py
# 负责数据的初始化，包括用户、服务器的分布，以及资源需求的生成。
import random
import numpy as np
# 固定 numpy 的随机种子
np.random.seed(42)
# 固定 random 模块的随机种子
random.seed(42)

# ========== 参数初始化 ==========
# 初始化用户、边缘服务器和云服务器的分布以及资源参数
# 初始化用户的资源需求


def initialize_topology(n, m_edge, m_cloud):
    """
    初始化用户和服务器分布，以及资源参数
    """
    # 用户位置和请求大小初始化
    user_positions = np.random.uniform(0, 100, (n, 2))  # 随机生成用户的二维坐标
    request_sizes = np.random.uniform(0.5, 12, n)  # 请求大小 (MB/s)

    # 用户优先级
    priorities = np.random.choice([1, 2, 3], size=n, p=[0.6, 0.3, 0.1])  # 用户优先级

    # 优先级权重
    priority_weights = {1: 1, 2: 2, 3: 3}  # 优先级到权重的映射
    weights = np.array([priority_weights[priority] for priority in priorities])  # 生成每个用户的权重

    priority_levels = {1: 1.0, 2: 1.2, 3: 1.5}
    levels = np.array([priority_levels[priority] for priority in priorities])  # 优先级的加权系数

    # 服务器位置初始化
    edge_positions = np.random.uniform(0, 100, (m_edge, 2))  # 边缘服务器的位置
    cloud_positions = np.random.uniform(100, 200, (m_cloud, 2))  # 云服务器的位置
    server_positions = np.vstack([edge_positions, cloud_positions])  # 合并服务器位置

    # 用户资源分配(不同优先级根据权重分配)
    # 根据请求大小计算带宽需求 (单位：Mbps)
    bandwidth_demands = request_sizes * 1  # 假设每MB请求需要1Mbps带宽
    bandwidth_demands = bandwidth_demands * levels
    # 根据请求大小计算CPU需求 (单位：CPU)
    cpu_demands = np.random.uniform(0.2, 0.4, n) * request_sizes  # 假设每MB请求需要0.2~0.4个CPU
    cpu_demands = cpu_demands * levels
    # 根据请求大小计算内存需求 (单位：GB)
    mem_demands = np.random.uniform(0.02, 0.1, n) * request_sizes  # 假设每MB请求需要0.01~0.05GB内存
    mem_demands = mem_demands * levels

    # 服务器资源上限
    R_cpu = np.concatenate([
        np.random.randint(20, 30, m_edge),  # 边缘服务器 CPU 核数
        np.random.randint(60, 90, m_cloud)  # 云服务器 CPU 核数
    ])
    R_mem = np.concatenate([
        np.random.randint(4, 8, m_edge),  # 边缘服务器内存（GB）
        np.random.randint(16, 32, m_cloud)  # 云服务器内存（GB）
    ])
    R_bandwidth = np.concatenate([
        np.random.randint(80, 100, m_edge),  # 边缘服务器带宽（Mbps）
        np.random.randint(200, 500, m_cloud)  # 云服务器带宽（Mbps）
    ])

    return user_positions, request_sizes, priorities, weights, server_positions, R_cpu, R_mem, R_bandwidth, cpu_demands, \
        mem_demands, bandwidth_demands
