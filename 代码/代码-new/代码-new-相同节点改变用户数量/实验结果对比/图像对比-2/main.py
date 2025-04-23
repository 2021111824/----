import os
import matplotlib.pyplot as plt
import numpy as np

# 设置支持中文的字体和字体大小
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体字体，也可以换成其他支持中文的字体，如 'SimSun'（宋体）
plt.rcParams['font.size'] = 10  # 设置全局字体大小
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 输出结果文件夹
output_folder = "simulation_results"
os.makedirs(output_folder, exist_ok=True)


# 对比不同实验的不同优先级用户的平均响应时间
def comparison_response_times():
    experiment_1 = [8.13, 4.65, 3.70]  # 贪心
    experiment_2 = [8.11, 4.28, 2.81]  # 优化求解器
    experiment_3 = [6.57, 5.45, 4.34]  # GA
    experiment_4 = [5.15, 4.96, 4.65]  # 无公平性

    # 类别名称
    categories = ['Priority 1', 'Priority 2', 'Priority 3']

    # 设置柱状图的宽度
    bar_width = 0.2

    # 每组实验的 x 轴位置（保证每组数据的柱子不重叠）
    index = np.arange(len(categories))

    # 创建图形并设置轴
    fig, ax = plt.subplots()

    experiments = [experiment_1, experiment_2, experiment_3, experiment_4]
    labels = ['贪心', '优化求解器', 'GA', '无公平性']
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
    plt.savefig(image_path, transparent=False)
    # 显示图形
    plt.show()


# 对比不同实验的总平均响应时间
def comparison_average_time():
    experiment_1 = [6.50]  # 贪心
    experiment_2 = [6.25]  # 优化求解器
    experiment_3 = [5.92]  # GA
    experiment_4 = [5.02]  # 无公平性

    # 类别名称
    categories = ['Different Experiments']

    # 设置柱状图的宽度
    bar_width = 0.05
    # 定义柱子之间的间距倍数
    gap_multiplier = 1.2

    # 每组实验的 x 轴位置（保证每组数据的柱子不重叠）
    index = np.arange(len(categories))

    # 创建图形并设置轴
    fig, ax = plt.subplots()

    experiments = [experiment_1, experiment_2, experiment_3, experiment_4]
    labels = ['贪心', '优化求解器', 'GA', '无公平性']
    colors = ['#ffbf4c', '#4c4cff', '#a64ca6', 'gray']

    for i, experiment in enumerate(experiments):
        # 调整柱子的位置，增加间距
        bars = ax.bar(index + (i - 1.5) * bar_width * gap_multiplier, experiment, bar_width, label=labels[i],
                      color=colors[i])
        # 标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom',
                    fontsize=10)

    # 添加标签、标题和图例
    ax.set_ylabel('Average Response Time(ms)')
    ax.set_title('Comparison of Average Response Time Across Different Experiments')
    ax.set_xticks(index)
    ax.set_xticklabels(categories)
    ax.legend()

    # 保存图片到指定文件夹
    image_path = os.path.join(output_folder, 'Average Response Time.png')
    plt.savefig(image_path, transparent=False)
    # 显示图形
    plt.show()


# 对比不同实验的Jain公平性指数
def comparison_jain_fairness():
    experiment_1 = [0.9756]  # 贪心
    experiment_2 = [0.9889]  # 优化求解器
    experiment_3 = [0.7236]  # GA
    experiment_4 = [0.6640]  # 无公平性

    # 类别名称
    categories = ['Different Experiments']

    # 设置柱状图的宽度
    bar_width = 0.05
    # 定义柱子之间的间距倍数
    gap_multiplier = 1.2

    # 每组实验的 x 轴位置（保证每组数据的柱子不重叠）
    index = np.arange(len(categories))

    # 创建图形并设置轴
    fig, ax = plt.subplots()

    experiments = [experiment_1, experiment_2, experiment_3, experiment_4]
    labels = ['贪心', '优化求解器', 'GA', '无公平性']
    colors = ['#ffbf4c', '#4c4cff', '#a64ca6', 'gray']

    for i, experiment in enumerate(experiments):
        # 调整柱子的位置，增加间距
        bars = ax.bar(index + (i - 1.5) * bar_width * gap_multiplier, experiment, bar_width, label=labels[i],
                      color=colors[i])
        # 标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom',
                    fontsize=10)

    # 添加标签、标题和图例
    ax.set_ylabel('Jain-fairness-index')
    ax.set_title('Comparison of Jain-fairness-index Across Different Experiments')
    ax.set_xticks(index)
    ax.set_xticklabels(categories)
    ax.legend()

    # 保存图片到指定文件夹
    image_path = os.path.join(output_folder, 'Jain-fairness-index.png')
    plt.savefig(image_path, transparent=False)
    # 显示图形
    plt.show()


if __name__ == "__main__":
    comparison_response_times()
    comparison_jain_fairness()
    comparison_average_time()
