# 初始化模块--initialize.py
# 负责数据的初始化，包括用户、服务器的分布，以及资源需求的生成。
import random
import numpy as np
# 固定 numpy 的随机种子
np.random.seed(40)
# 固定 random 模块的随机种子
random.seed(40)


# ========== 参数初始化 ==========
# 初始化用户、边缘服务器和云服务器的分布以及资源参数
# 初始化用户的资源需求
def initialize_topology():
    # ========== 1. 用户数据 ==========
    # 用户数量初始化
    n = 140

    # 用户位置初始化
    user_positions = np.random.uniform(0, 200, (n, 2))  # 随机生成用户的二维坐标，在（0,100）坐标区域

    # 用户优先级初始化
    priorities = np.random.choice([1, 2, 3], size=n, p=[0.6, 0.3, 0.1])  # 用户优先级

    # 用户权重初始化
    priority_weights = {1: 1, 2: 1.5, 3: 2}  # 优先级到权重的映射
    weights = np.array([priority_weights[priority] for priority in priorities])  # 生成每个用户的权重

    # 用户请求服务数据大小初始化(MB)
    data_in = np.random.uniform(0.5, 1, n)
    data_out = np.random.uniform(0.5, 1, n)
    user_data = (data_in + data_out) * 8  # 单位转换 1B = 8bit

    # 用户请求服务的计算单位量(CU)
    p_user = np.random.uniform(12, 15, n)

    # 用户的计算能力分配量(CU/ms)
    P_allocation = weights * p_user * np.random.uniform(0.08, 0.12)

    # 各优先级用户的最大响应时间
    T_max = {
        1: 30,  # 优先级 1 用户最大允许响应时间 (ms)
        2: 25,  # 优先级 2 用户最大允许响应时间 (ms)
        3: 20,  # 优先级 3 用户最大允许响应时间 (ms)
    }

    # ========== 2. 服务器数据 ==========
    # 服务器数量初始化
    m_edge = 10  # 边缘服务器数
    m_cloud = 1  # 云服务器数

    # 服务器位置初始化
    edge_positions = np.random.uniform(0, 200, (m_edge, 2))  # 边缘服务器的位置
    cloud_positions = np.random.uniform(200, 300, (m_cloud, 2))  # 云服务器的位置
    server_positions = np.vstack([edge_positions, cloud_positions])  # 合并服务器位置

    # 传播时延初始化
    # 用户与边缘服务器之间的延迟矩阵
    t_delay_e = np.random.uniform(1, 3, (n, m_edge + m_cloud))
    t_delay_c = np.random.uniform(8, 10, n)

    # 服务器总可用带宽初始化
    R_bandwidth = np.concatenate([
        np.random.uniform(100000, 200000, m_edge),  # 边缘服务器总可用带宽为 5-10Gbps
        np.full(m_cloud, 800)  # 云服务器可用带宽为 500Mbps
    ])
    # print(R_bandwidth)

    # 服务器可用计算资源初始化
    R_edge = np.concatenate([
        np.random.uniform(250, 300, m_edge),  # 边缘服务器的可用计算资源(RU)
        np.full(m_cloud, 200000)  # 云服务器计算资源丰富
    ])

    # 边缘服务器的总计算能力初始化
    P_edge = R_edge

    # 云服务计算能力初始化
    P_cloud = 4  # 单位（CU/ms）

    # ========== 3. 服务实例数据 ==========
    # 单个服务实例的计算能力
    p_m = 10  # 单位（CU/ms）

    # 部署一个服务实例需要的计算资源
    r_m = 10  # 单位（RU）

    # ========== 4. 成本参数 ==========
    monthly_fixed_cost = 60  # 每月固定成本（单位：某种货币）
    daily_fixed_cost = monthly_fixed_cost / 30  # 每日固定成本
    cost_edge = {"fixed": daily_fixed_cost}  # 边缘服务器成本单价
    cost_cloud = {"p_net": 0.5}  # 云服务器成本单价
    max_cost = 100000  # 最大允许总成本

    # 返回初始化数据
    return n, user_positions, priorities, weights, user_data, p_user, P_allocation, T_max, \
        m_edge, m_cloud, server_positions, t_delay_e, t_delay_c, R_bandwidth, R_edge, P_edge, P_cloud, \
        p_m, r_m, cost_edge, cost_cloud, max_cost

