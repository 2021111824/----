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
def initialize_topology():
    # ========== 1. 参数设置 ==========
    n, m_edge, m_cloud = 150, 20, 3  # 用户数、边缘服务器数、云服务器数
    v_edge, v_cloud = 10, 5  # 边缘服务器和云服务器的网络传播速度 (Mbps)
    P_edge, P_cloud = 300, 3000  # 边缘和云服务器的计算能力 (MB/s)
    T_max = {
        1: 10,  # 优先级 1 用户最大允许响应时间 (ms)
        2: 8,  # 优先级 2 用户最大允许响应时间 (ms)
        3: 6,  # 优先级 3 用户最大允许响应时间 (ms)
    }

    # 成本参数
    monthly_fixed_cost = 20  # 每月固定成本（单位：某种货币）
    daily_fixed_cost = monthly_fixed_cost / 30  # 每日固定成本
    cost_edge = {"fixed": daily_fixed_cost, "cpu": 0.5, "mem": 0.3, "bandwidth": 0.1}  # 边缘服务器成本
    cost_cloud = {"cpu": 0.8, "mem": 0.5, "bandwidth": 0.2}  # 云服务器成本
    p_net = 0.5  # 网络流量单位成本
    max_cost = 1000  # 最大允许总成本

    # ========== 2. 生成用户数据 ==========
    # 用户位置和请求大小初始化
    user_positions = np.random.uniform(0, 100, (n, 2))  # 随机生成用户的二维坐标，在（0,100）坐标区域

    # 用户优先级
    priorities = np.random.choice([1, 2, 3], size=n, p=[0.6, 0.3, 0.1])  # 用户优先级

    # 优先级权重
    priority_weights = {1: 1, 2: 2, 3: 3}  # 优先级到权重的映射
    weights = np.array([priority_weights[priority] for priority in priorities])  # 生成每个用户的权重

    # 资源分配系数
    priority_levels = {1: 1.0, 2: 1.2, 3: 1.5}
    levels = np.array([priority_levels[priority] for priority in priorities])  # 优先级的加权系数

    # 用户类型划分：40% 低需求，40% 中等需求，20% 高需求
    user_types = np.random.choice(["low", "medium", "high"], size=n, p=[0.4, 0.4, 0.2])

    request_sizes = np.random.uniform(0.5, 12, n)  # 请求大小 (MB/s)

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

    # ========== 3. 生成服务器数据 ==========
    # 服务器位置初始化
    edge_positions = np.random.uniform(0, 100, (m_edge, 2))  # 边缘服务器的位置
    cloud_positions = np.random.uniform(100, 200, (m_cloud, 2))  # 云服务器的位置
    server_positions = np.vstack([edge_positions, cloud_positions])  # 合并服务器位置

    # 服务器的资源调整
    R_cpu = np.concatenate([
        np.random.randint(8, 32, m_edge),  # 边缘服务器 CPU 8~32 核
        np.random.randint(64, 256, m_cloud)  # 云服务器 CPU 64~256 核
    ])

    R_mem = np.concatenate([
        np.random.randint(8, 64, m_edge),  # 边缘服务器内存 8~64 GB
        np.random.randint(64, 1024, m_cloud)  # 云服务器内存 64~1024 GB
    ])

    R_bandwidth = np.concatenate([
        np.random.randint(100, 1000, m_edge),  # 边缘服务器带宽 100~1000 Mbps
        np.random.randint(1000, 5000, m_cloud)  # 云服务器带宽 10Gbps~50Gbps
    ])

    return n, m_edge, m_cloud, v_edge, v_cloud, P_edge, P_cloud,\
        T_max, cost_edge, cost_cloud, p_net, max_cost, \
        user_positions, request_sizes, priorities, weights, server_positions, \
        R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands
