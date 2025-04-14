import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 输出结果文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # Example data
    execution_ratios = [7, 8, 9, 10]  # 横坐标
    jian_indexes = {
        # 'Gurobi': [0.9939, 0.9925, 0.9923, 0.9940],
        # 'GSAFO': [0.9935, 0.9916, 0.9917, 0.9745],
        'GSAFO': [0.9599, 0.9900, 0.9886, 0.9876],
        'GA': [0.9584, 0.9790, 0.9883, 0.9820],
        'TABU': [0.9572, 0.9644, 0.9835, 0.9733],
        'Greedy': [0.9438, 0.9890, 0.9768, 0.9181],
        'Random': [0.8647, 0.8693, 0.8480, 0.8441]
    }

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    bar_width = 0.1
    index = np.arange(len(execution_ratios))

    colors = ['#86b573', '#e4c286', '#bb7f7e', '#a6c2f1', '#4f7b99']  # Colors slightly darker

    for i, (label, jain_index) in enumerate(jian_indexes.items()):
        ax.bar(index + i * bar_width, jain_index, bar_width, label=label, color=colors[i])

    ax.set_xlabel('Number of Servers')
    ax.set_ylabel('Jain fairness index')
    ax.set_title('Jain fairness index of Different Algorithms')

    # 设置横坐标标签为 7, 8, 9, 10
    ax.set_xticks(index + bar_width * 2.5)
    ax.set_xticklabels([f'{x}' for x in execution_ratios])

    # 设置纵坐标从0.75开始，每次增大0.05
    ax.set_ylim(0.75, 1.0)  # 设置纵坐标范围
    ax.set_yticks(np.arange(0.75, 1.00, 0.05))  # 设置纵坐标刻度

    # 调整图例位置，放置在图像下方
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

    # 调整布局
    plt.tight_layout()
    # 保存图片到指定文件夹
    image_path = os.path.join(output_folder, 'Jain-fairness-index-compair.png')
    plt.savefig(image_path, transparent=False)
    plt.show()

