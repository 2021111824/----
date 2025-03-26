import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from initialize import initialize_topology
from calculations import assign_computational_capacity, compute_response_time, calculate_response_stats
from visualization import (
    save_priority_distribution,
    plot_user_server_distribution,
    plot_response_time_distribution,
    plot_avg_response_time,
    plot_server_resource_usage,
    plot_cost_distribution,
    plot_user_server_connections,
    save_user_server_mapping
)


# 创建优化模型
def create_model():
    # ========== 数据初始化 ==========
    n, m_edge, m_cloud, v_edge, v_cloud, b_edge, b_cloud, P_edge, P_cloud, \
        T_max, cost_edge, cost_cloud, p_net, max_cost, \
        user_positions, request_sizes, priorities, weights, server_positions, \
        R_cpu, R_mem, R_bandwidth, cpu_demands, mem_demands, bandwidth_demands = initialize_topology()

    # 创建 Gurobi 模型
    model = gp.Model("Maximize_Weighted_Jain")

    # 设置求解参数
    model.setParam('FeasibilityTol', 1e-9)  # 可行性容差
    model.setParam('MIPFocus', 1)  # 1 优先寻找可行解；2 证明当前解最优
    model.setParam('MIPGap', 0.01)  # 最优性差距容忍度
    model.setParam('TimeLimit', 3000)  # 时间限制设置为 3000 秒
    model.setParam('Threads', 4)  # 求解器使用的线程数
    model.setParam('Presolve', 1)  # 开启预求解

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
                user_positions[i], server_positions[j], is_edge, request_sizes[i],
                assign_computational_capacity(individual, user_positions, server_positions, request_sizes,
                                              P_edge, P_cloud, m_edge, priorities)[i],
                v_edge, v_cloud, b_edge, b_cloud
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

    # 添加约束2：服务器资源约束
    for j in range(m_edge + m_cloud):
        # CPU资源约束
        model.addConstr(gp.quicksum(cpu_demands[i] * x[i, j] for i in range(n)) <= R_cpu[j], name=f"cpu_constraint_{j}")
        # 内存资源约束
        model.addConstr(gp.quicksum(mem_demands[i] * x[i, j] for i in range(n)) <= R_mem[j], name=f"mem_constraint_{j}")
        # 带宽资源约束
        model.addConstr(gp.quicksum(bandwidth_demands[i] * x[i, j] for i in range(n)) <= R_bandwidth[j],
                        name=f"bandwidth_constraint_{j}")

    # 添加约束3：每个优先级的所有用户的平均响应时间不超过该优先级的最大响应时间
    user_capacities = assign_computational_capacity(
        individual, user_positions, server_positions, request_sizes, P_edge, P_cloud, m_edge, priorities
    )
    for level in np.unique(priorities):
        idx = np.where(priorities == level)[0]
        max_response_time_for_user_priority = T_max[level]
        model.addConstr(
            gp.quicksum(
                compute_response_time(user_positions[i], server_positions[j], j < m_edge,
                                      request_sizes[i], user_capacities[i], v_edge, v_cloud, b_edge, b_cloud) * x[i, j]
                for i in idx for j in range(m_edge + m_cloud)
            ) <= max_response_time_for_user_priority * len(idx),
            name=f"response_time_constraint_{level}"
        )

    # 添加约束4：总成本不能超过最大成本
    total_cost_expr = gp.quicksum(
        (
            (cost_edge["cpu"] * cpu_demands[i] + cost_edge["mem"] * mem_demands[i] + cost_edge["bandwidth"] *
             bandwidth_demands[i] + cost_edge["fixed"]) * x[i, j] if j < m_edge else
            (cost_cloud["cpu"] * cpu_demands[i] + cost_cloud["mem"] * mem_demands[i] + cost_cloud["bandwidth"] *
             bandwidth_demands[i] + p_net * request_sizes[i]) * x[i, j]
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
    return model, x, individual, user_positions, request_sizes, priorities, weights, server_positions, R_cpu, R_mem, R_bandwidth, \
        cpu_demands, mem_demands, bandwidth_demands, cost_edge, cost_cloud,  n, m_edge, m_cloud, P_edge, P_cloud, \
        v_edge, v_cloud, b_edge, b_cloud, T_max, p_net, max_cost


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
    model, x, individual, user_positions, request_sizes, priorities, weights, server_positions, R_cpu, R_mem, R_bandwidth, \
        cpu_demands, mem_demands, bandwidth_demands, cost_edge, cost_cloud,  n, m_edge, m_cloud, P_edge, P_cloud, \
        v_edge, v_cloud, b_edge, b_cloud, T_max, p_net, max_cost = create_model()

    # 求解模型
    model.optimize()

    # 检查求解状态
    if model.status == GRB.OPTIMAL:
        # 重置 individual 为全零
        final_individual = np.zeros((n, m_edge + m_cloud), dtype=int)

        # 解析结果
        for i in range(n):
            for j in range(m_edge + m_cloud):
                if x[i, j].x > 0.5:
                    final_individual[i, j] = 1

        user_capacities = assign_computational_capacity(
            final_individual, user_positions, server_positions, request_sizes, P_edge, P_cloud, m_edge, priorities
        )

        # 根据分配情况计算每个用户的响应时间
        response_times = []
        for u in range(n):
            server_idx = np.argmax(final_individual[u])
            is_edge = server_idx < m_edge
            response_time = compute_response_time(
                user_positions[u], server_positions[server_idx], is_edge, request_sizes[u],
                user_capacities[u], v_edge, v_cloud, b_edge, b_cloud
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
            f.write(f"Jain公平性指数： {F_jain:.4f}\n")
            f.write(f"所有用户的平均响应时间: {avg_response_time:.2f} ms\n\n")
            f.write(f"Response Time Statistics by Priority:\n")
            for level, stats in response_stats.items():
                status = "OK" if stats["mean"] <= T_max[level] else "EXCEEDED"
                f.write(f"  Priority {level}: Mean={stats['mean']:.2f} ms (Limit: {T_max[level]} ms) [{status}]\n")

        print(f"Simulation results saved to '{output_folder}'.\n")

        # 计算各服务器的资源使用情况
        server_cpu_usage = np.sum(final_individual * cpu_demands[:, None], axis=0)
        server_mem_usage = np.sum(final_individual * mem_demands[:, None], axis=0)
        server_bandwidth_usage = np.sum(final_individual * bandwidth_demands[:, None], axis=0)

        # 获取分配到云服务器的用户索引
        user_allocated_to_cloud = np.where(np.argmax(final_individual, axis=1) >= m_edge)[0]

        # 计算成本分布
        cost_details = {
            "edge": {
                "cpu": np.sum(server_cpu_usage[:m_edge] * cost_edge["cpu"]),
                "mem": np.sum(server_mem_usage[:m_edge] * cost_edge["mem"]),
                "bandwidth": np.sum(server_bandwidth_usage[:m_edge] * cost_edge["bandwidth"]),
                "fixed": m_edge * cost_edge["fixed"]
            },
            "cloud": {
                "cpu": np.sum(server_cpu_usage[m_edge:] * cost_cloud["cpu"]),
                "mem": np.sum(server_mem_usage[m_edge:] * cost_cloud["mem"]),
                "bandwidth": np.sum(server_bandwidth_usage[m_edge:] * cost_cloud["bandwidth"]),
                "network": np.sum(request_sizes[user_allocated_to_cloud] * p_net)
            },
        }
        total_edge_cost = sum(cost_details["edge"].values())
        total_cloud_cost = sum(cost_details["cloud"].values())
        total_cost = total_edge_cost + total_cloud_cost

        # 可视化和其他分析
        save_user_server_mapping(final_individual, output_folder)
        save_priority_distribution(priorities, output_folder)
        plot_user_server_distribution(user_positions, server_positions, priorities, m_edge, output_folder)
        plot_user_server_connections(user_positions, server_positions, final_individual, priorities, m_edge, output_folder)
        plot_response_time_distribution(response_times, priorities, output_folder)
        plot_avg_response_time(response_times, priorities, output_folder, T_max)
        plot_server_resource_usage(server_cpu_usage, server_mem_usage, server_bandwidth_usage,
                                   R_cpu, R_mem, R_bandwidth, m_edge, output_folder)
        plot_cost_distribution(cost_details, output_folder, total_edge_cost, total_cloud_cost, total_cost, max_cost)
    else:
        print(f"求解失败，状态码: {model.status}")


if __name__ == "__main__":
    solve_model()