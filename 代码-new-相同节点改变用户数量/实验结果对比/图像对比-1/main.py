import os

import matplotlib.pyplot as plt
import numpy as np

# 输出结果文件夹
output_folder = "simulation_results"
os.makedirs(output_folder, exist_ok=True)


# 对比不同实验的不同优先级用户的平均响应时间
def comparison_response_times():
    experiment_1 = [7.95, 6.02, 4.54]  # NSGA - II，三个优先级用户的平均响应时间
    experiment_2 = [7.87, 5.98, 4.46]  # fairness-GA
    experiment_3 = [9.21, 8.83, 5.67]  # MILP
    experiment_4 = [4.39, 4.88, 4.61]  # non-fairness

    # 类别名称
    categories = ['Priority 1', 'Priority 2', 'Priority 3']

    # 设置柱状图的宽度
    bar_width = 0.2

    # 每组实验的 x 轴位置（保证每组数据的柱子不重叠）
    index = np.arange(len(categories))

    # 创建图形并设置轴
    fig, ax = plt.subplots()

    experiments = [experiment_1, experiment_2, experiment_3, experiment_4]
    labels = ['NSGA - II', 'fairness-GA', 'MILP', 'non-fairness']
    colors = ['#ffbf4c', '#4c4cff', '#a64ca6', 'gray']

    for i, experiment in enumerate(experiments):
        bars = ax.bar(index + (i - 1.5) * bar_width, experiment, bar_width, label=labels[i], color=colors[i])
        # 标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

    # 添加标签、标题和图例
    ax.set_ylabel('Response Time (ms)')
    ax.set_title('Comparison of Response Times Across Different Experiments')
    ax.set_xticks(index)
    ax.set_xticklabels(categories)
    ax.legend()

    # 保存图片到指定文件夹
    image_path = os.path.join(output_folder, 'response_time_comparison.png')
    plt.savefig(image_path)
    # 显示图形
    plt.show()


# 对比不同实验的Jain公平性指数
def comparison_jain_fairness():
    experiment_1 = [0.6330, 0.8918, 0.9613]  # NSGA - II
    experiment_2 = [0.6007, 0.8657, 0.9547]  # fairness-GA
    experiment_3 = [0.4922, 0.5125, 0.8114]  # MILP
    experiment_4 = [0.7857, 0.7980, 0.7696]  # non-fairness

    # 类别名称
    categories = ['Priority 1', 'Priority 2', 'Priority 3']

    # 设置柱状图的宽度
    bar_width = 0.2

    # 每组实验的 x 轴位置（保证每组数据的柱子不重叠）
    index = np.arange(len(categories))

    # 创建图形并设置轴
    fig, ax = plt.subplots()

    experiments = [experiment_1, experiment_2, experiment_3, experiment_4]
    labels = ['NSGA - II', 'fairness-GA', 'MILP', 'non-fairness']
    colors = ['#ffbf4c', '#4c4cff', '#a64ca6', 'gray']

    for i, experiment in enumerate(experiments):
        bars = ax.bar(index + (i - 1.5) * bar_width, experiment, bar_width, label=labels[i], color=colors[i])
        # 标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom', fontsize=6)

    # 添加标签、标题和图例
    ax.set_ylabel('Jain - fairness - index')
    ax.set_title('Comparison of Jain - fairness - index Across Different Experiments')
    ax.set_xticks(index)
    ax.set_xticklabels(categories)
    ax.legend()

    # 保存图片到指定文件夹
    image_path = os.path.join(output_folder, 'Jain - fairness - index.png')
    plt.savefig(image_path)
    # 显示图形
    plt.show()


# 对比不同实验的响应时间比例偏差
def comparison_response_deviation():
    experiment_1 = [0.0528]  # NSGA - II
    experiment_2 = [0.0413]  # fairness-GA
    experiment_3 = [0.5175]  # MILP
    experiment_4 = [0.7408]  # non-fairness

    # 类别名称
    categories = ['Different Experiments']

    # 设置柱状图的宽度
    bar_width = 0.2

    # 每组实验的 x 轴位置（保证每组数据的柱子不重叠）
    index = np.arange(len(categories))

    # 创建图形并设置轴
    fig, ax = plt.subplots()

    experiments = [experiment_1, experiment_2, experiment_3, experiment_4]
    labels = ['NSGA - II', 'fairness-GA', 'MILP', 'non-fairness']
    colors = ['#ffbf4c', '#4c4cff', '#a64ca6', 'gray']

    for i, experiment in enumerate(experiments):
        bars = ax.bar(index + (i - 1.5) * bar_width, experiment, bar_width, label=labels[i], color=colors[i])
        # 标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom',
                    fontsize=6)

    # 添加标签、标题和图例
    ax.set_ylabel('JResponse Deviation')
    ax.set_title('Comparison of Response Deviation Across Different Experiments')
    ax.set_xticks(index)
    ax.set_xticklabels(categories)
    ax.legend()

    # 保存图片到指定文件夹
    image_path = os.path.join(output_folder, 'Response Deviation.png')
    plt.savefig(image_path)
    # 显示图形
    plt.show()


if __name__ == "__main__":
    comparison_response_times()
    comparison_jain_fairness()
    comparison_response_deviation()