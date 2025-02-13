# 修复模块--repair.py
import numpy as np
from calculations import compute_response_time, calculate_response_stats  # 引入响应时间计算函数


def repair_individual(individual, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                      cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                      request_sizes, v_edge, v_cloud, b_edge, b_cloud, m_edge, m_cloud):
    """
    修复不满足约束的个体，确保每个用户都最终有服务器分配。
    优先级高的用户会优先连接到边缘服务器，若边缘服务器资源不足则连接到云服务器。
    普通用户根据距离和资源约束重新分配。
    """
    n_users, n_servers = individual.shape

    # 初始化资源使用情况
    server_cpu_usage = np.zeros(n_servers)
    server_mem_usage = np.zeros(n_servers)
    server_bandwidth_usage = np.zeros(n_servers)

    # 更新资源使用情况
    for i in range(n_users):
        server_idx = np.argmax(individual[i])  # 当前分配的服务器
        server_cpu_usage[server_idx] += cpu_demands[i]
        server_mem_usage[server_idx] += mem_demands[i]
        server_bandwidth_usage[server_idx] += bandwidth_demands[i]

    # 计算每个用户的响应时间
    response_times = [
        compute_response_time(
            user_positions[i], server_positions[np.argmax(individual[i])],
            np.argmax(individual[i]) < m_edge, request_sizes[i],
            v_edge, v_cloud, b_edge, b_cloud, individual, cpu_demands, mem_demands, bandwidth_demands, R_cpu, R_mem, R_bandwidth, m_edge, m_cloud, user_positions
        ) for i in range(n_users)
    ]

    # 持续修复，直到满足所有约束
    while True:
        # 记录是否进行了修复
        any_repair = False

        # 检查每个用户是否满足约束
        for i in range(n_users):
            server_idx = np.argmax(individual[i])  # 当前分配的服务器
            priority = priorities[i]

            # 计算该优先级的所有用户的平均响应时间
            avg_response_time = np.mean([response_times[j] for j in range(n_users) if priorities[j] == priority])

            # 如果优先级较高的用户不满足约束条件，首先尝试连接到距离较近的边缘服务器
            if priority == max(priorities):  # 优先级最高的用户
                if server_cpu_usage[server_idx] > R_cpu[server_idx] or \
                        server_mem_usage[server_idx] > R_mem[server_idx] or \
                        server_bandwidth_usage[server_idx] > R_bandwidth[server_idx] or \
                        avg_response_time > T_max[priority]:

                    # 清除当前分配
                    individual[i, server_idx] = 0
                    server_cpu_usage[server_idx] -= cpu_demands[i]
                    server_mem_usage[server_idx] -= mem_demands[i]
                    server_bandwidth_usage[server_idx] -= bandwidth_demands[i]

                    # 尝试重新分配到较近的边缘服务器
                    valid_edge_servers = [
                        new_server_idx for new_server_idx in range(m_edge)
                        if (server_cpu_usage[new_server_idx] + cpu_demands[i] <= R_cpu[new_server_idx] and
                            server_mem_usage[new_server_idx] + mem_demands[i] <= R_mem[new_server_idx] and
                            server_bandwidth_usage[new_server_idx] + bandwidth_demands[i] <= R_bandwidth[
                                new_server_idx])
                    ]

                    if valid_edge_servers:
                        # 计算每个边缘服务器到当前用户的距离
                        user_position = user_positions[i]
                        edge_distances = [
                            np.linalg.norm(user_position - server_positions[new_server_idx])  # 使用欧几里得距离
                            for new_server_idx in valid_edge_servers
                        ]

                        # 选择距离最近的边缘服务器
                        closest_server_idx = valid_edge_servers[np.argmin(edge_distances)]

                        # 更新分配
                        individual[i, closest_server_idx] = 1
                        server_cpu_usage[closest_server_idx] += cpu_demands[i]
                        server_mem_usage[closest_server_idx] += mem_demands[i]
                        server_bandwidth_usage[closest_server_idx] += bandwidth_demands[i]

                        # 重新计算当前用户的响应时间
                        response_times[i] = compute_response_time(
                            user_positions[i], server_positions[closest_server_idx],
                            closest_server_idx < m_edge, request_sizes[i],
                            v_edge, v_cloud, b_edge, b_cloud, individual, cpu_demands, mem_demands, bandwidth_demands, R_cpu, R_mem, R_bandwidth, m_edge, m_cloud, user_positions
                        )

                        any_repair = True
                    else:
                        # 如果没有可用的边缘服务器，则尝试云服务器
                        valid_cloud_servers = [
                            new_server_idx for new_server_idx in range(m_edge, n_servers)
                            if (server_cpu_usage[new_server_idx] + cpu_demands[i] <= R_cpu[new_server_idx] and
                                server_mem_usage[new_server_idx] + mem_demands[i] <= R_mem[new_server_idx] and
                                server_bandwidth_usage[new_server_idx] + bandwidth_demands[i] <= R_bandwidth[
                                    new_server_idx])
                        ]

                        if valid_cloud_servers:
                            # 选择一个有效的云服务器
                            new_server_idx = np.random.choice(valid_cloud_servers)

                            # 更新分配
                            individual[i, new_server_idx] = 1
                            server_cpu_usage[new_server_idx] += cpu_demands[i]
                            server_mem_usage[new_server_idx] += mem_demands[i]
                            server_bandwidth_usage[new_server_idx] += bandwidth_demands[i]

                            # 重新计算当前用户的响应时间
                            response_times[i] = compute_response_time(
                                user_positions[i], server_positions[new_server_idx],
                                new_server_idx < m_edge, request_sizes[i],
                                v_edge, v_cloud, b_edge, b_cloud, individual, cpu_demands, mem_demands, bandwidth_demands, R_cpu, R_mem, R_bandwidth, m_edge, m_cloud, user_positions
                            )

                            any_repair = True

            else:  # 普通优先级用户
                if server_cpu_usage[server_idx] > R_cpu[server_idx] or \
                   server_mem_usage[server_idx] > R_mem[server_idx] or \
                   server_bandwidth_usage[server_idx] > R_bandwidth[server_idx] or \
                   avg_response_time > T_max[priority]:

                    # 清除当前分配
                    individual[i, server_idx] = 0
                    server_cpu_usage[server_idx] -= cpu_demands[i]
                    server_mem_usage[server_idx] -= mem_demands[i]
                    server_bandwidth_usage[server_idx] -= bandwidth_demands[i]

                    # 尝试重新分配到离用户较近的服务器
                    valid_servers = [
                        new_server_idx for new_server_idx in range(n_servers)
                        if (server_cpu_usage[new_server_idx] + cpu_demands[i] <= R_cpu[new_server_idx] and
                            server_mem_usage[new_server_idx] + mem_demands[i] <= R_mem[new_server_idx] and
                            server_bandwidth_usage[new_server_idx] + bandwidth_demands[i] <= R_bandwidth[new_server_idx])
                    ]

                    if valid_servers:
                        # 选择一个有效的服务器
                        new_server_idx = np.random.choice(valid_servers)

                        # 更新分配
                        individual[i, new_server_idx] = 1
                        server_cpu_usage[new_server_idx] += cpu_demands[i]
                        server_mem_usage[new_server_idx] += mem_demands[i]
                        server_bandwidth_usage[new_server_idx] += bandwidth_demands[i]

                        # 重新计算当前用户的响应时间
                        response_times[i] = compute_response_time(
                            user_positions[i], server_positions[new_server_idx],
                            new_server_idx < m_edge, request_sizes[i],
                            v_edge, v_cloud, b_edge, b_cloud, individual, cpu_demands, mem_demands, bandwidth_demands, R_cpu, R_mem, R_bandwidth, m_edge, m_cloud, user_positions
                        )

                        any_repair = True

        # 如果没有进行任何修复，跳出循环
        if not any_repair:
            break

    # 重新计算每个优先级的平均响应时间
    response_stats = calculate_response_stats(response_times, priorities)

    # 如果平均响应时间没有超限，退出修复
    for priority, stats in response_stats.items():
        avg_response_time = stats["mean"]
        if avg_response_time > T_max[priority]:
            # 如果修复后仍未满足约束，则继续修复
            return repair_individual(individual, user_positions, server_positions, R_cpu, R_mem, R_bandwidth,
                                     cpu_demands, mem_demands, bandwidth_demands, priorities, T_max,
                                     request_sizes, v_edge, v_cloud, b_edge, b_cloud, m_edge, m_cloud)

    return individual  # 返回修复后的个体
