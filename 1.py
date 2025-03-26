import numpy as np
import pandas as pd

# 初始化参数
num_edge_servers = 10
num_cloud_servers = 1
num_users = 100

# 边缘服务器参数
edge_server_capacity = np.random.randint(30, 60, num_edge_servers)  # RU
edge_server_bandwidth = np.random.randint(800, 1000, num_edge_servers)  # Mbps
edge_latency = 10  # ms

# 云服务器参数
cloud_server_capacity = 10000  # CU/ms
cloud_server_bandwidth = np.random.randint(100, 200)  # Mbps
cloud_latency = 100  # ms

# 微服务实例参数
service_capacity_per_instance = 100  # CU/ms
instance_resource_usage = 5  # RU

# 用户请求参数
user_data = pd.DataFrame({
    "user_id": range(num_users),
    "D_in": np.random.randint(1, 10, num_users),  # 1-10 MB
    "W_i": np.random.uniform(1, 3, num_users),  # 用户权重 1-3
    "p_i": np.random.randint(5, 30, num_users)  # 计算需求 5-30 CU
})

# 计算 D_out
user_data["D_out"] = user_data["D_in"] * np.random.uniform(0.8, 1.2, num_users)

# 计算用户带宽需求
user_data["bandwidth_needed"] = user_data["D_in"] * user_data["W_i"] * 8  # Mbps

# 随机分配用户到边缘服务器
user_data["assigned_edge_server"] = np.random.randint(0, num_edge_servers, num_users)

# 计算边缘服务器总计算能力
service_instances_per_edge = edge_server_capacity // instance_resource_usage
edge_processing_capacity = service_instances_per_edge * service_capacity_per_instance

# 计算用户的计算资源分配
user_data["P_ij"] = (user_data["p_i"] * user_data["W_i"]) / (
    user_data.groupby("assigned_edge_server")["p_i"].transform("sum") * user_data["W_i"]
) * edge_processing_capacity[user_data["assigned_edge_server"]]

# 计算传输延迟
user_data["edge_transmission_latency"] = (user_data["D_in"] + user_data["D_out"]) / np.random.choice(edge_server_bandwidth, num_users) * 1000  # ms
user_data["cloud_transmission_latency"] = (user_data["D_in"] + user_data["D_out"]) / cloud_server_bandwidth * 1000  # ms

# 计算处理延迟
def compute_latency(p, P):
    return p / P if P > 0 else float('inf')

user_data["edge_processing_latency"] = user_data.apply(lambda row: compute_latency(row["p_i"], row["P_ij"]), axis=1)
user_data["cloud_processing_latency"] = user_data["p_i"] / cloud_server_capacity

# 计算总响应时间
user_data["edge_response_time"] = user_data["edge_transmission_latency"] + user_data["edge_processing_latency"] + edge_latency
user_data["cloud_response_time"] = user_data["cloud_transmission_latency"] + user_data["cloud_processing_latency"] + cloud_latency

# 计算 Jain 公平性指数
def compute_jain_fairness(response_times, weights):
    numerator = np.sum(response_times * weights) ** 2
    denominator = len(response_times) * np.sum((response_times * weights) ** 2)
    return numerator / denominator

jain_fairness_edge = compute_jain_fairness(user_data["edge_response_time"], user_data["W_i"])
jain_fairness_cloud = compute_jain_fairness(user_data["cloud_response_time"], user_data["W_i"])

# 计算最大支持用户数
max_edge_users = np.sum(edge_server_capacity) // instance_resource_usage
max_cloud_users = cloud_server_capacity // 30
max_supported_users = min(max_edge_users + max_cloud_users, num_users)

# 输出结果
print("边缘服务器响应时间范围:", user_data["edge_response_time"].min(), "-", user_data["edge_response_time"].max(), "ms")
print("云服务器响应时间范围:", user_data["cloud_response_time"].min(), "-", user_data["cloud_response_time"].max(), "ms")
print("Jain 公平性指数（边缘）:", jain_fairness_edge)
print("Jain 公平性指数（云）:", jain_fairness_cloud)
print("最大支持用户数:", max_supported_users)
