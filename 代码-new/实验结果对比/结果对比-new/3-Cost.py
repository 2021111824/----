import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 输出结果文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # Example data
    execution_ratios = [140, 160, 180, 200]  # 修改横坐标为 140, 160, 180, 200
    costs = {
        'Gurobi': [241.38, 226.93, 214.28, 321.43],
        # 'GSAFO': [213.97, 214.22, 200.00, 271.67],
        'GSAFO': [200.00, 226.40, 200.00, 296.06],
        'GA': [226.05, 264.79, 345.68, 489.52],
        'Greedy': [238.97, 225.85, 212.04, 333.76],
        'Random': [373.08, 454.84, 444.56, 489.66]
    }

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    bar_width = 0.1
    index = np.arange(len(execution_ratios))

    colors = ['#bb7f7e', '#e4c286', '#86b573', '#a6c2f1', '#4f7b99']  # Colors slightly darker

    for i, (label, cost) in enumerate(costs.items()):
        ax.bar(index + i * bar_width, cost, bar_width, label=label, color=colors[i])

    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Cost')
    ax.set_title('Total Cost')

    # 设置横坐标标签为 140, 160, 180, 200
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

