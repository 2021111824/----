import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 输出结果文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # Example data
    execution_ratios = [140, 160, 180, 200]  # 修改横坐标为 140, 160, 180, 200
    execution_times = {
        'Gurobi': [15.14, 17.70, 30.75, 28.83],
        'GSAFO': [109.03, 136.59, 162.43, 181.43],
        'GA': [140.31, 286.80, 428.70, 701.14],
        'Greedy': [9.10, 13.77, 14.80, 17.16],
        'Random': [1.71, 1.47, 1.77, 1.74]
    }

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    bar_width = 0.1
    index = np.arange(len(execution_ratios))

    colors = ['#bb7f7e', '#e4c286', '#86b573', '#a6c2f1', '#4f7b99']  # Colors slightly darker

    for i, (label, execution_time) in enumerate(execution_times.items()):
        ax.bar(index + i * bar_width, execution_time, bar_width, label=label, color=colors[i])

    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Execution Time(ms)')
    ax.set_title('Execution Time of Different Algorithms')

    # 设置横坐标标签为 140, 160, 180, 200
    ax.set_xticks(index + bar_width * 2.5)
    ax.set_xticklabels([f'{x}' for x in execution_ratios])

    # 设置纵坐标从0开始，每次增大
    ax.set_ylim(0, 800)  # 设置纵坐标范围
    ax.set_yticks(np.arange(0, 800, 100))  # 设置纵坐标刻度

    # 调整图例位置，放置在图像下方
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

    # 调整布局
    plt.tight_layout()
    # 保存图片到指定文件夹
    image_path = os.path.join(output_folder, 'Execution_Time.png')
    plt.savefig(image_path, transparent=False)
    plt.show()

