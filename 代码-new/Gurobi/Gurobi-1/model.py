import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from initialize import initialize_topology
from calculations import assign_bandwidth_capacity, compute_response_time, calculate_response_stats, \
    calculate_total_cost
from visualization import (
    save_priority_distribution,
    plot_user_server_distribution,
    plot_response_time_distribution,
    plot_avg_response_time,
    plot_server_resource_usage,
    plot_cost_distribution,
    plot_user_server_connections,
    plot_service_instance_distribution
)


# 创建优化模型
def create_model():
    # ========== 数据初始化 ==========
    n, user_positions, priorities, weights, user_data, p_user, P_allocation, T_max, \
        m_edge, m_cloud, server_positions, t_delay_e, t_delay_c, R_bandwidth, R_edge, P_edge, P_cloud, \
        p_m, r_m, cost_edge, cost_cloud, max_cost = initialize_topology()

    # 创建 Gurobi 模型
    model = gp.Model("Maximize_Weighted_Jain")

    # 设置求解参数
    model.setParam('FeasibilityTol', 1e-9)  # 可行性容差
    model.setParam('MIPFocus', 2)  # 1 优先寻找可行解；2 证明当前解最优
    model.setParam('MIPGap', 0.01)  # 最优性差距容忍度
    model.setParam('TimeLimit', 30000)  # 时间限制设置为 3000 秒
    model.setParam('Threads', 4)  # 求解器使用的线程数
    model.setParam('Presolve', 0)  # 开启预求解

    # 创建决策变量--表示用户 i 是否连接到服务器 j
    x = {}
    for i in range(n):
        for j in range(m_edge + m_cloud):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"user_{i}_server_{j}")

    # 引入辅助变量 S 和 Q，分别用于计算 加权响应时间总和 和 加权响应时间平方和
    S = model.addVar(vtype=GRB.CONTINUOUS, name="S")
    Q = model.addVar(vtype=GRB.CONTINUOUS, lb=1e-6, name="Q")  # 为避免除零问题，设置 Q 的下界
    # 引入新的辅助变量 Z 来近似目标函数--实现加权Jain最大化
    Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

    # 初始化用户分配
    individual = np.zeros((n, m_edge + m_cloud), dtype=int)

    # 计算每个用户 - 服务器组合的响应时间和加权响应时间
    response_times = {}
    weighted_response_times = {}
    for i in range(n):
        for j in range(m_edge + m_cloud):
            is_edge = j < m_edge  # 判断是否是边缘服务器
            response_time = compute_response_time(
                t_delay_e[i][j], t_delay_c[i], is_edge, user_data[i],
                assign_bandwidth_capacity(individual, n, m_edge, m_cloud, user_data, R_bandwidth)[i],
                p_user[i], P_allocation[i]
            )
            weighted_response_time = response_time * weights[i]  # 加权响应时间
            response_times[(i, j)] = response_time
            weighted_response_times[(i, j)] = weighted_response_time

    # 设置目标函数：最大化 Z
    model.setObjective(Z, sense=GRB.MAXIMIZE)

    # 添加约束：Z <= S^2 / (n * Q) 的近似约束
    # 这里我们可以通过乘以 n * Q 来避免除法，得到 n * Q * Z <= S^2
    model.addConstr(n * Q * Z <= S * S, name="objective_approx_constraint")

    # 添加约束1：每个用户只能连接到一个服务器
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(m_edge + m_cloud)) == 1, name=f"user_{i}_connection")

    # 添加约束2：边缘服务器资源约束
    for j in range(m_edge):
        # 计算每个服务器上的计算能力需求
        server_compute_capability = gp.quicksum(p_user[i] * x[i, j] for i in range(n))  # 每个服务器上的计算能力需求

        # 定义整数变量表示每台服务器上服务实例的数量
        server_instance_num = model.addVar(vtype=GRB.INTEGER, name=f"server_instance_num_{j}")

        # 添加约束：确保 server_instance_num 是 server_compute_capability / p_m 的上界（取整）
        model.addConstr(server_instance_num * p_m >= server_compute_capability, name=f"server_compute_capability_{j}")
        model.addConstr(server_instance_num >= server_compute_capability / p_m,
                        name=f"server_compute_capability_lower_bound_{j}")

        # 计算服务器的计算资源使用情况
        server_compute_resource_usage = server_instance_num * r_m

        # 添加约束：服务器资源使用不能超过服务器的最大资源
        model.addConstr(server_compute_resource_usage <= R_edge[j], name=f"server_resource_limit_{j}")

    # 添加约束3：每个优先级的所有用户的平均响应时间不超过该优先级的最大响应时间
    user_bandwidth = assign_bandwidth_capacity(
        individual, n, m_edge, m_cloud, user_data, R_bandwidth
    )
    for level in np.unique(priorities):
        idx = np.where(priorities == level)[0]
        max_response_time_for_user_priority = T_max[level]
        model.addConstr(
            gp.quicksum(
                compute_response_time(t_delay_e[i][j], t_delay_c[i], j < m_edge, user_data[i], user_bandwidth[i], p_user[i], P_allocation[i]) * x[i, j]
                for i in idx for j in range(m_edge + m_cloud)
            ) <= max_response_time_for_user_priority * len(idx),
            name=f"response_time_constraint_{level}"
        )

    # 添加约束4：总成本不能超过最大成本
    total_cost_expr = gp.quicksum(
        (
            (cost_edge["fixed"]) * x[i, j] if j < m_edge else
            (cost_cloud["p_net"] * p_user[i]) * x[i, j]
        )
        for i in range(n) for j in range(m_edge + m_cloud)
    )
    model.addConstr(total_cost_expr <= max_cost, name="total_cost_constraint")

    # 添加约束：计算 S、Q
    # S 为所有用户加权响应时间的总和
    model.addConstr(
        gp.quicksum(weighted_response_times[(i, j)] * x[i, j] for i in range(n) for j in range(m_edge + m_cloud)) == S,
        name="S_constraint"
    )
    # Q 为所有用户加权响应时间平方的总和
    model.addConstr(
        gp.quicksum((weighted_response_times[(i, j)] ** 2) * x[i, j] for i in range(n) for j in range(m_edge + m_cloud))
        == Q,
        name="Q_constraint"
    )

    # 更新模型
    model.update()

    # 返回模型及必要的参数
    return model, x, individual, user_positions, priorities, weights, server_positions, R_bandwidth, \
        cost_edge, cost_cloud,  n, m_edge, m_cloud, \
        T_max, max_cost, user_data, t_delay_e, t_delay_c, p_user, P_allocation, p_m, r_m, R_edge


# 求解模型
def solve_model():
    """
    求解模型
    :return: 最终的用户 - 服务器分配矩阵
    """

    # 创建输出文件夹
    output_folder = "./visualization_results"
    os.makedirs(output_folder, exist_ok=True)

    # 创建模型
    model, x, individual, user_positions, priorities, weights, server_positions, R_bandwidth, \
        cost_edge, cost_cloud,  n, m_edge, m_cloud, \
        T_max, max_cost, user_data, t_delay_e, t_delay_c, p_user, P_allocation, p_m, r_m, R_edge = create_model()

    # 求解模型
    model.optimize()

    # 检查求解状态
    if model.status == GRB.OPTIMAL:
        best_objective_value = model.objVal  # 获取最优目标值
        # 重置 individual 为全零
        final_individual = np.zeros((n, m_edge + m_cloud), dtype=int)

        # 解析结果
        for i in range(n):
            for j in range(m_edge + m_cloud):
                if x[i, j].x > 0.5:
                    final_individual[i, j] = 1

        user_bandwidth = assign_bandwidth_capacity(
            final_individual, n, m_edge, m_cloud, user_data, R_bandwidth
        )

        # 根据分配情况计算每个用户的响应时间
        response_times = []
        for u in range(n):
            server_idx = np.argmax(final_individual[u])
            is_edge = server_idx < m_edge
            response_time = compute_response_time(
                t_delay_e[u][server_idx], t_delay_c[u], is_edge,
                user_data[u], user_bandwidth[u], p_user[u], P_allocation[u]
            )
            response_times.append(response_time)

        avg_response_time = np.mean(response_times)

        # 响应时间统计
        response_stats = calculate_response_stats(response_times, priorities)

        # 计算最终结果的Jain公平性指数
        response_times = np.array(response_times)
        weighted_times = np.array([response_times[i] * weights[i] for i in range(len(response_times))])
        F_jain = (np.sum(weighted_times) ** 2) / (len(weighted_times) * np.sum(weighted_times ** 2))

        # 保存详细结果到文件
        with open(os.path.join(output_folder, "simulation_results.txt"), "w") as f:
            f.write("===== Simulation Results =====\n")
            f.write(f"Jain公平性指数： {best_objective_value:.4f}\n")
            f.write(f"所有用户的平均响应时间: {avg_response_time:.2f} ms\n\n")
            f.write(f"Response Time Statistics by Priority:\n")
            for level, stats in response_stats.items():
                status = "OK" if stats["mean"] <= T_max[level] else "EXCEEDED"
                f.write(f"  Priority {level}: Mean={stats['mean']:.2f} ms (Limit: {T_max[level]} ms) [{status}]\n")

            f.write("\nUser-to-Server Assignment:\n")
            for i in range(len(user_positions)):
                server_idx = np.argmax(final_individual[i])
                server_type = "Edge" if server_idx < m_edge else "Cloud"
                f.write(f"  User {i} -> Server {server_idx} ({server_type})\n")
        print(f"Simulation results saved to '{output_folder}'.\n")

        # 记录资源使用情况
        server_compute_capability = np.zeros(len(server_positions))

        # 遍历用户，统计每台服务器的资源使用
        for i in range(len(user_positions)):
            server_idx = np.argmax(final_individual[i])  # 获取用户分配到的服务器
            is_edge = server_idx < m_edge  # 判断是否是边缘服务器

            # 使用用户的计算能力需求更新服务器计算能力需求
            server_compute_capability[server_idx] += p_user[i]  # 服务器上的用户计算能力需求

        # 服务器上的计算资源使用情况
        server_compute_resource_usage = (np.ceil(server_compute_capability / p_m)) * r_m  # 服务器的计算资源使用情况

        # 计算每个服务器上部署的服务实例数量
        service_instances = server_compute_resource_usage / r_m

        # 计算总成本和分项成本
        total_cost, cost_details = calculate_total_cost(final_individual, m_edge, cost_edge, cost_cloud, p_user)

        # 可视化和其他分析
        save_priority_distribution(priorities, output_folder)
        plot_user_server_distribution(user_positions, server_positions, priorities, m_edge, output_folder)
        # 2. 绘制响应时间分布
        plot_response_time_distribution(response_times, priorities, output_folder)

        # 3. 绘制平均响应时间柱状图
        plot_avg_response_time(response_times, priorities, output_folder, T_max)

        # 4. 绘制服务器资源使用情况
        plot_server_resource_usage(server_compute_resource_usage, R_edge, m_edge, output_folder)

        # 5. 绘制用户和服务器的连接图
        plot_user_server_connections(user_positions, server_positions, final_individual, priorities, m_edge, output_folder)

        # 6. 绘制服务器部署成本图
        plot_cost_distribution(cost_details, output_folder,
                               total_edge_cost=cost_details['edge']['fixed'],
                               total_cloud_cost=cost_details['cloud']['p_net'],
                               total_cost=total_cost,
                               cost_limit=max_cost)

        # 7. 绘制服务器上的服务实例部署情况
        plot_service_instance_distribution(service_instances, output_folder)
    else:
        print(f"求解失败，状态码: {model.status}")


if __name__ == "__main__":
    solve_model()