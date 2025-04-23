import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 输出结果文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # Example data
    execution_ratios = [120, 140, 160, 180, 200]  # 修改横坐标为 140, 160, 180, 200
    execution_times = {
        # 'Gurobi': [15.14, 17.70, 16.29, 28.83],
        # 'GSAFO': [109.03, 136.59, 159.29, 181.43],
        'FCGDO': [24.19, 33.96, 40.19, 39.56, 40.63],
        'GA': [71.10, 152.60, 337.01, 460.56, 768.39],
        'Optimal': [33.67, 45.33, 55.29, 67.39, 71.99],
        'Greedy': [7.34, 10.76, 11.36, 14.33, 14.44],
        'Random': [2.57, 1.92, 1.90, 1.88, 1.67]
    }

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    bar_width = 0.1
    index = np.arange(len(execution_ratios))

    colors = ['#86b573', '#e4c286', '#bb7f7e', '#a6c2f1', '#4f7b99']  # Colors slightly darker

    for i, (label, execution_time) in enumerate(execution_times.items()):
        ax.bar(index + i * bar_width, execution_time, bar_width, label=label, color=colors[i])

    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Execution Time(s)')
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

