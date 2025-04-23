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
    n = 150

    # 用户位置初始化
    # 80%的用户在中心区域，20%的用户在较远区域
    user_positions = np.zeros((n, 2))

    # 80%用户在中心区域（靠近边缘服务器）
    center_users = np.random.choice(np.arange(n), size=int(n * 0.8), replace=False)
    user_positions[center_users] = np.random.uniform(-100, 100, (len(center_users), 2))  # 用户位置在（0, 200）区域

    # 20%用户在远离边缘服务器的区域
    far_users = np.setdiff1d(np.arange(n), center_users)
    user_positions[far_users] = np.random.uniform(-200, 200, (len(far_users), 2))  # 用户位置在（200, 300）区域

    # 用户优先级初始
    priorities = np.random.choice([1, 2, 3], size=n, p=[0.6, 0.3, 0.1])

    # 用户权重初始化
    priority_weights = {1: 1, 2: 1.5, 3: 2}
    weights = np.array([priority_weights[priority] for priority in priorities])

    # 用户请求服务数据大小初始化(MB)
    data_in = np.random.uniform(0.5, 1, n)
    data_out = np.random.uniform(0.5, 1, n)
    user_data = (data_in + data_out) * 8  # 单位转换 1B = 8bit

    # 用户请求服务的计算单位量(CU)
    p_user = np.random.uniform(1000, 1200, n)

    # 用户的计算能力分配量(CU/ms)
    P_allocation = weights * p_user * np.random.uniform(0.008, 0.012)

    # 各优先级用户的最大响应时间
    T_max = {
        1: 200,
        2: 150,
        3: 100,
    }

    # ========== 2. 服务器数据 ==========
    m_edge = 11  # 边缘服务器数
    m_cloud = 1  # 云服务器数

    # 服务器位置初始化
    edge_positions = np.random.uniform(-100, 100, (m_edge, 2))  # 边缘服务器的位置
    cloud_positions = np.random.uniform(300, 400, (m_cloud, 2))  # 云服务器的位置
    server_positions = np.vstack([edge_positions, cloud_positions])  # 合并服务器位置

    # 传播时延初始化
    # 计算用户到服务器的传播延迟
    t_delay_e = np.zeros((n, m_edge + m_cloud))
    for i in range(n):
        for j in range(m_edge):
            dist = np.linalg.norm(user_positions[i] - edge_positions[j])  # 计算用户和边缘服务器的距离
            if dist < 100:
                t_delay_e[i, j] = np.random.uniform(1, 5)  # 中心用户到边缘服务器延迟 1-5ms
            else:
                t_delay_e[i, j] = np.random.uniform(8, 10)  # 边缘的用户到边缘服务器延迟 5-10ms

    t_delay_c = np.random.uniform(20, 40, n)

    # 服务器总可用带宽初始化
    R_bandwidth = np.concatenate([
        np.random.uniform(10000, 20000, m_edge),  # 边缘服务器总可用带宽为 10-20Gbps
        np.full(m_cloud, 100)  # 云服务器可用带宽为 100Mbps
    ])
    # print(R_bandwidth)

    # 服务器可用计算资源初始化
    R_edge = np.concatenate([
        np.random.uniform(300, 500, m_edge),  # 边缘服务器的可用计算资源(RU)
        np.full(m_cloud, 200000)  # 云服务器计算资源丰富
    ])

    # 边缘服务器的总计算能力初始化
    P_edge = R_edge

    # 云服务器计算能力初始化
    P_cloud = 40  # 单位（CU/ms）

    # ========== 3. 服务实例数据 ==========
    # 单个服务实例的计算能力
    p_m = 20  # 单位（CU/ms）

    # 部署一个服务实例需要的计算资源
    r_m = 20  # 单位（RU）

    # ========== 4. 成本参数 ==========
    monthly_fixed_cost = 6000  # 每月固定成本（单位：元）
    daily_fixed_cost = monthly_fixed_cost / 30  # 每日固定成本
    cost_edge = {"fixed": daily_fixed_cost}  # 边缘服务器成本单价
    cost_cloud = {"p_net": 0.1}  # 云服务器成本单价
    max_cost = 8000  # 最大允许总成本

    # ========== 5. 遗传算法参数 ==========
    Population = 400  # 初始种群大小
    G_max = 400  # 最大迭代代数
    P_crossover, P_mutation = 0.8, 0.2  # 交叉概率和变异概率

    # 返回初始化数据
    return n, user_positions, priorities, weights, user_data, p_user, P_allocation, T_max, \
        m_edge, m_cloud, server_positions, t_delay_e, t_delay_c, R_bandwidth, R_edge, P_edge, P_cloud,\
        p_m, r_m, cost_edge, cost_cloud, max_cost, Population, G_max, P_crossover, P_mutation

