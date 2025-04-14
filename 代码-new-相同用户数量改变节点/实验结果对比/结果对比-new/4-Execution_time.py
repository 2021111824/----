import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 输出结果文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # Example data
    execution_ratios = [7, 8, 9, 10]  # 横坐标
    execution_times = {
        # 'Gurobi': [9.02, 8.36, 10.89, 12.19],
        # 'GSAFO': [109.03, 136.59, 159.29, 181.43],
        'GSAFO': [21.58, 26.55, 32.14, 39.43],
        'GA': [378.93, 302.76, 229.61, 143.30],
        'TABU': [184.89, 150.81, 179.50, 209.02],
        'Greedy': [6.52, 7.18, 8.47, 10.67],
        'Random': [1.81, 1.81, 1.71, 1.61]
    }

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    bar_width = 0.1
    index = np.arange(len(execution_ratios))

    colors = ['#86b573', '#e4c286', '#bb7f7e', '#a6c2f1', '#4f7b99']

    for i, (label, execution_time) in enumerate(execution_times.items()):
        ax.bar(index + i * bar_width, execution_time, bar_width, label=label, color=colors[i])

    ax.set_xlabel('Number of Servers')
    ax.set_ylabel('Execution Time(s)')
    ax.set_title('Execution Time of Different Algorithms')

    # 设置横坐标标签为 7, 8, 9, 10
    ax.set_xticks(index + bar_width * 2.5)
    ax.set_xticklabels([f'{x}' for x in execution_ratios])

    # 设置纵坐标从0开始，每次增大
    ax.set_ylim(0, 400)  # 设置纵坐标范围
    ax.set_yticks(np.arange(0, 400, 50))  # 设置纵坐标刻度

    # 调整图例位置，放置在图像下方
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

    # 调整布局
    plt.tight_layout()
    # 保存图片到指定文件夹
    image_path = os.path.join(output_folder, 'Execution_Time.png')
    plt.savefig(image_path, transparent=False)
    plt.show()

