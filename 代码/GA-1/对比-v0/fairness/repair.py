# 修复模块--repair.py
import numpy as np
from calculations import compute_response_time  # 引入响应时间计算函数


def repair_individual(individual, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                      cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                      request_sizes, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, m_edge):
    """
    修复不满足约束的个体。

    Args:
        individual (ndarray): 用户到服务器的分配矩阵
        user_positions (ndarray): 用户位置
        server_positions (ndarray): 服务器位置
        R_cpu, R_mem, R_bandwidth (ndarray): 每个服务器的资源限制
        cpu_demands, mem_demands, bandwidth_demands (ndarray): 每个用户的资源需求
        priorities (ndarray): 用户优先级
        T_max (dict): 每个优先级的最大平均响应时间约束，格式为{priority: max_time}

    Returns:
        ndarray: 修复后的个体
    """
    n_users, n_servers = individual.shape

    # 初始化资源使用统计
    server_cpu_usage = np.zeros(n_servers)
    server_mem_usage = np.zeros(n_servers)
    server_bandwidth_usage = np.zeros(n_servers)

    # 更新资源使用情况
    for i in range(n_users):
        server_idx = np.argmax(individual[i])  # 当前分配的服务器
        server_cpu_usage[server_idx] += cpu_demands[i]
        server_mem_usage[server_idx] += mem_demands[i]
        server_bandwidth_usage[server_idx] += bandwidth_demands[i]

    # 遍历用户，修复分配
    for i in range(n_users):
        server_idx = np.argmax(individual[i])

        # 检查当前分配是否违反资源限制
        if (server_cpu_usage[server_idx] > R_cpu[server_idx] or
                server_mem_usage[server_idx] > R_mem[server_idx] or
                server_bandwidth_usage[server_idx] > R_bandwidth[server_idx]):

            # 清除当前分配
            individual[i, server_idx] = 0
            server_cpu_usage[server_idx] -= cpu_demands[i]
            server_mem_usage[server_idx] -= mem_demands[i]
            server_bandwidth_usage[server_idx] -= bandwidth_demands[i]

            # 尝试重新分配到符合资源限制的服务器
            valid_servers = []

            for new_server_idx in range(n_servers):
                if (server_cpu_usage[new_server_idx] + cpu_demands[i] <= R_cpu[new_server_idx] and
                        server_mem_usage[new_server_idx] + mem_demands[i] <= R_mem[new_server_idx] and
                        server_bandwidth_usage[new_server_idx] + bandwidth_demands[i] <= R_bandwidth[new_server_idx]):
                    valid_servers.append(new_server_idx)

            # 随机从符合条件的服务器中选择一个
            if valid_servers:
                new_server_idx = np.random.choice(valid_servers)

                # 计算新的响应时间
                new_response_time = compute_response_time(
                    user_positions[i], server_positions[new_server_idx],
                    new_server_idx < m_edge, request_sizes[i],
                    v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud
                )

                # 检查新的响应时间和资源需求
                if (new_response_time <= T_max[priorities[i]] and
                        server_cpu_usage[new_server_idx] + cpu_demands[i] <= R_cpu[new_server_idx] and
                        server_mem_usage[new_server_idx] + mem_demands[i] <= R_mem[new_server_idx] and
                        server_bandwidth_usage[new_server_idx] + bandwidth_demands[i] <= R_bandwidth[new_server_idx]):
                    # 如果响应时间和资源使用都符合要求
                    individual[i, new_server_idx] = 1
                    server_cpu_usage[new_server_idx] += cpu_demands[i]
                    server_mem_usage[new_server_idx] += mem_demands[i]
                    server_bandwidth_usage[new_server_idx] += bandwidth_demands[i]

    return individual  # 在修复完所有用户后返回修复后的个体
