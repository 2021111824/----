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
    # 参数配置
    n, m_edge, m_cloud = 150, 20, 3  # 用户数、边缘服务器数、云服务器数
    v_edge, v_cloud = 10, 5  # 边缘服务器和云服务器的网络传播速度 (Mbps)
    b_edge, b_cloud = 100, 500  # 边缘和云服务器的带宽速度  (MB/s)
    P_edge, P_cloud = 500, 1000  # 边缘和云服务器的计算能力 (MB/s)
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

    # 用户位置和请求大小初始化
    user_positions = np.random.uniform(0, 100, (n, 2))  # 随机生成用户的二维坐标
    request_sizes = np.random.uniform(0.5, 12, n)  # 请求大小 (MB/s)

    # 用户优先级
    priorities = np.random.choice([1, 2, 3], size=n, p=[0.6, 0.3, 0.1])  # 用户优先级

    # 优先级权重
    priority_weights = {1: 1, 2: 2, 3: 3}  # 优先级到权重的映射
    weights = np.array([priority_weights[priority] for priority in priorities])  # 生成每个用户的权重

    priority_levels = {1: 1, 2: 1.2, 3: 1.5}
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

    return n, m_edge, m_cloud, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, \
        T_max, cost_edge, cost_cloud, p_net, max_cost, \
        user_positions, request_sizes, priorities, weights, server_positions, \
        R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands
