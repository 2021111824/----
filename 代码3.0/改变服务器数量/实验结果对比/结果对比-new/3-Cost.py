import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 输出结果文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # Example data
    execution_ratios = [7, 8, 9, 10]  # 横坐标
    costs = {
        # 'Gurobi': [343.26, 200.45, 247.46, 213.15],
        # 'GSAFO': [213.97, 214.22, 200.00, 271.67],
        'GSAFO': [339.68, 160.00, 194.44, 200.00],
        'GA': [484.37, 377.61, 207.98, 253.20],
        'Optimal': [370.34, 294.71, 246.18, 295.67],
        'Greedy': [357.71, 160.00, 208.24, 353.33],
        'Random': [437.03, 415.23, 384.37, 439.61]
    }

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    bar_width = 0.1
    index = np.arange(len(execution_ratios))

    colors = ['#86b573', '#e4c286', '#bb7f7e', '#a6c2f1', '#4f7b99']

    for i, (label, cost) in enumerate(costs.items()):
        ax.bar(index + i * bar_width, cost, bar_width, label=label, color=colors[i])

    ax.set_xlabel('Number of Servers')
    ax.set_ylabel('Cost')
    ax.set_title('Total Cost of Different Algorithms')

    # 设置横坐标标签为 7, 8, 9, 10
    ax.set_xticks(index + bar_width * 2.5)
    ax.set_xticklabels([f'{x}' for x in execution_ratios])

    # 设置纵坐标从0开始，每次增大0.05
    ax.set_ylim(0, 500)  # 设置纵坐标范围
    ax.set_yticks(np.arange(0, 500, 50))  # 设置纵坐标刻度

    # 调整图例位置，放置在图像下方
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

    # 调整布局
    plt.tight_layout()
    # 保存图片到指定文件夹
    image_path = os.path.join(output_folder, 'Total_cost.png')
    plt.savefig(image_path, transparent=False)
    plt.show()

