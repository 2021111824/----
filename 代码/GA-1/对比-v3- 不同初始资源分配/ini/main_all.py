import json


from ini.initialize import initialize_topology
import os


def initialize_simulation_data(n, m_edge, m_cloud):
    # 获取初始化的拓扑数据
    user_positions, request_sizes, priorities, weights, server_positions, R_cpu, R_mem, R_bandwidth, \
        cpu_demands, mem_demands, bandwidth_demands = initialize_topology(n, m_edge, m_cloud)
    edge_positions = server_positions[:m_edge]
    cloud_positions = server_positions[m_edge:]

    # 将初始化数据封装到字典中
    data = {
        'user_positions': user_positions.tolist(),  # 如果是numpy数组，转换为列表以便json存储
        'request_sizes': request_sizes.tolist(),
        'priorities': priorities.tolist(),
        'weights': weights.tolist(),
        'server_positions': server_positions.tolist(),
        'R_cpu': R_cpu.tolist(),
        'R_mem': R_mem.tolist(),
        'R_bandwidth': R_bandwidth.tolist(),
        'cpu_demands': cpu_demands.tolist(),
        'mem_demands': mem_demands.tolist(),
        'bandwidth_demands': bandwidth_demands.tolist(),
        'edge_positions': edge_positions.tolist(),
        'cloud_positions': cloud_positions.tolist()
    }

    # 保存数据到文件（例如json格式）
    file_name = 'simulation_data.json'
    if os.path.exists(file_name):
        print(f"警告: {file_name} 文件已经存在，将被覆盖！")

    try:
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"数据已成功保存到 {file_name}")
    except Exception as e:
        print(f"保存数据时发生错误: {e}")

    return data


if __name__ == "__main__":
    # ========= 参数设置 ==========
    n, m_edge, m_cloud = 200, 20, 3  # 用户数、边缘服务器数、云服务器数
    simulation_data = initialize_simulation_data(n, m_edge, m_cloud)
