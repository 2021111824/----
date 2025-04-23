import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 输出结果文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # Example data
    execution_ratios = [110, 130, 150, 170, 190]  # 修改横坐标为 110, 130, 150, 170, 190
    execution_times = {
        # 'Gurobi': [15.14, 17.70, 16.29, 28.83],
        # 'GSAFO': [109.03, 136.59, 159.29, 181.43],
        'FCGDO': [19.70, 27.26, 34.66, 41.17, 39.06],
        'GA': [72.58, 122.50, 208.56, 363.17, 567.07],
        'Optimal': [27.84, 38.58, 49.97, 62.49, 71.33],
        'Greedy': [6.47, 8.10, 10.34, 12.96, 14.40],
        'Random': [1.62, 1.72, 1.54, 2.11, 1.67]
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

