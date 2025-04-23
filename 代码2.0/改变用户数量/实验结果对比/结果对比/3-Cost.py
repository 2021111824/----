import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 输出结果文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # Example data
    execution_ratios = [120, 140, 160, 180, 200]  # 修改横坐标为 140, 160, 180, 200
    costs = {
        # 'Gurobi': [241.38, 226.93, 214.28, 321.43],
        'FCGDO': [2000.00, 2000.00, 2000.00, 2000.00, 4352.11],
        'GA': [2818.54, 2551.37, 2656.80, 4573.78, 4763.92],
        'Optimal': [2277.25, 2655.97, 2924.34, 2915.88, 4139.52],
        'Greedy': [2000.00, 2663.32, 2136.94, 2148.41, 4087.05],
        'Random': [3502.83, 3855.50, 4425.45, 3906.49, 4891.57]
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
    ax.set_ylim(0, 5000)  # 设置纵坐标范围
    ax.set_yticks(np.arange(0, 5000, 500))  # 设置纵坐标刻度

    # 调整图例位置，放置在图像下方
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

    # 调整布局
    plt.tight_layout()
    # 保存图片到指定文件夹
    image_path = os.path.join(output_folder, 'Total_cost.png')
    plt.savefig(image_path, transparent=False)
    plt.show()

