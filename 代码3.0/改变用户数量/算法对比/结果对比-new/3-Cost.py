import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 输出结果文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # Example data
    execution_ratios = [100, 120, 140, 160, 180, 200]
    costs = {
        # 'Gurobi': [0.9957, 0.9928, 0.9948, 0.9908],
        # 'GSAFO': [0.9935, 0.9916, 0.9917, 0.9745],
        'FCGDO': [2000.00, 2000.00, 2000.00, 2000.00, 2000.00, 3778.15],
        'GA': [2269.82, 2817.57, 2927.22, 2952.80, 2557.20, 4545.79],
        'Optimal': [2799.70, 2963.36, 3573.69, 3596.46, 4262.14, 5280.09],
        'Greedy': [3264.76, 2963.36, 4007.83, 4742.82, 4775.74, 5461.83],
        'Random': [3602.58, 3502.83, 3855.50, 4425.45, 4458.45, 5374.60]
    }

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    bar_width = 0.1
    index = np.arange(len(execution_ratios))

    colors = ['#86b573', '#e4c286', '#bb7f7e', '#a6c2f1', '#4f7b99']  # Colors slightly darker

    for i, (label, cost) in enumerate(costs.items()):
        ax.bar(index + i * bar_width, cost, bar_width, label=label, color=colors[i])

    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Cost')
    ax.set_title('Total Cost of Different Algorithms')

    # 设置横坐标标签为 140, 160, 180, 200
    ax.set_xticks(index + bar_width * 2.5)
    ax.set_xticklabels([f'{x}' for x in execution_ratios])

    # 设置纵坐标从0开始，每次增大0.05
    ax.set_ylim(0, 6000)  # 设置纵坐标范围
    ax.set_yticks(np.arange(0, 6000, 500))  # 设置纵坐标刻度

    # 调整图例位置，放置在图像下方
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

    # 调整布局
    plt.tight_layout()
    # 保存图片到指定文件夹
    image_path = os.path.join(output_folder, 'Total_cost.png')
    plt.savefig(image_path, transparent=False)
    plt.show()

