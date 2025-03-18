# 初始化模块--initialize.py
# 负责数据的初始化，包括用户、服务器的分布，以及资源需求的生成。
import random
import numpy as np
# 固定 numpy 的随机种子
# np.random.seed(42)
# 固定 random 模块的随机种子
# random.seed(42)


# ========== 参数初始化 ==========
# 初始化用户、边缘服务器和云服务器的分布以及资源参数
# 初始化用户的资源需求
def initialize_topology():
    # ========== 1. 参数设置 ==========
    n, m_edge, m_cloud = 300, 5, 2  # 用户数、边缘服务器数、云服务器数
    t_delay_e,  t_delay_c = 0.01, 0.1  # 边缘和云服务器的传播时延（s)
    P_edge, P_cloud = 1000, 10000  # 边缘和云服务器的计算能力 (MB/s)
    T_max = {
        1: 150,  # 优先级 1 用户最大允许响应时间 (s)
        2: 120,  # 优先级 2 用户最大允许响应时间 (s)
        3: 100,  # 优先级 3 用户最大允许响应时间 (s)
    }

    # 成本参数
    monthly_fixed_cost = 20  # 每月固定成本（单位：某种货币）
    daily_fixed_cost = monthly_fixed_cost / 30  # 每日固定成本
    cost_edge = {"fixed": daily_fixed_cost, "compute": 0.5, "bandwidth": 0.01}  # 边缘服务器成本
    cost_cloud = {"p_net": 0.5, "compute": 1.0, "bandwidth": 0.02}  # 云服务器成本
    max_cost = 1000  # 最大允许总成本

    # ========== 2. 生成用户数据 ==========
    # 用户位置和请求大小初始化
    user_positions = np.random.uniform(0, 200, (n, 2))  # 随机生成用户的二维坐标，在（0,100）坐标区域

    # 用户优先级
    priorities = np.random.choice([1, 2, 3], size=n, p=[0.6, 0.3, 0.1])  # 用户优先级

    # 优先级权重
    priority_weights = {1: 1, 2: 1.2, 3: 1.5}  # 优先级到权重的映射
    weights = np.array([priority_weights[priority] for priority in priorities])  # 生成每个用户的权重

    # 资源分配系数
    priority_levels = {1: 1, 2: 1.2, 3: 1.5}
    levels = np.array([priority_levels[priority] for priority in priorities])  # 优先级的加权系数

    # 生成请求大小 (MB/s)
    request_sizes = np.random.uniform(1, 10, n)
    print(request_sizes)

    # 计算资源需求
    compute_demands = np.random.uniform(0.1, 0.2, n) * request_sizes  # 每MB请求需要0.05-0.1个单位的计算资源
    compute_demands = compute_demands * levels

    # 计算带宽需求（Mbps）
    bandwidth_demands = np.random.uniform(1, 2, n) * request_sizes * 8
    # bandwidth_demands = request_sizes * 1 * 8  # 每MB请求需要1Mbps * 8 带宽
    bandwidth_demands = bandwidth_demands * levels

    # ========== 3. 生成服务器数据 ==========
    # 服务器位置初始化
    edge_positions = np.random.uniform(0, 200, (m_edge, 2))  # 边缘服务器的位置
    cloud_positions = np.random.uniform(200, 300, (m_cloud, 2))  # 云服务器的位置
    server_positions = np.vstack([edge_positions, cloud_positions])  # 合并服务器位置

    # 服务器的计算资源
    # 边缘服务器与云服务器的总计算资源量都是
    R_compute = np.concatenate([
        np.full(m_edge, 100),
        np.full(m_cloud, 10000)
    ])

    # 服务器的带宽上限
    R_bandwidth = np.concatenate([
        np.full(m_edge, 5000),  # 边缘服务器总可用带宽为 5 Gbps
        np.full(m_cloud, 1000)  # 云服务器可用带宽为 1 Gbps
    ])

    return n, m_edge, m_cloud, t_delay_e,  t_delay_c, P_edge, P_cloud,\
        T_max, cost_edge, cost_cloud, max_cost, \
        user_positions, request_sizes, priorities, weights, server_positions, \
        R_compute, R_bandwidth, compute_demands, bandwidth_demands
