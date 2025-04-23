import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 输出结果文件夹
    output_folder = "simulation_results"
    os.makedirs(output_folder, exist_ok=True)

    # Example data
    execution_ratios = [140, 160, 180, 200]

    response_times_1 = {
        'Gurobi': [15.93, 13.69, 12.47, 13.44],
        # 'GSAFO': [15.86, 13.34, 15.00, 12.73],
        'GSAFO': [15.78, 13.77, 12.08, 12.77],
        'GA': [15.63, 13.30, 12.21, 13.27],
        'Greedy': [15.85, 13.25, 11.98, 12.84],
        'Random': [16.30, 14.21, 13.57, 13.84]
    }

    response_times_2 = {
        'Gurobi': [11.80, 9.85, 8.62, 9.13],
        # 'GSAFO': [11.66, 9.73, 11.21, 9.51],
        'GSAFO': [11.68, 10.01, 9.17, 10.19],
        'GA': [11.35, 9.75, 8.92, 9.29],
        'Greedy': [11.63, 10.22, 9.26, 10.40],
        'Random': [13.13, 11.80, 10.31, 10.48]
    }

    response_times_3 = {
        'Gurobi': [8.87, 7.77, 7.14, 7.63],
        # 'GSAFO': [9.76, 7.98, 9.26, 8.26],
        'GSAFO': [9.61, 8.56, 7.52, 8.11],
        'GA': [9.25, 8.30, 7.68, 7.80],
        'Greedy': [9.65, 8.15, 7.86, 8.56],
        'Random': [9.33, 11.45, 8.15, 8.87]
    }

    # Plotting the line chart
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define line styles for each data set
    line_styles = ['-', '--', ':']  # Solid line, dashed line, dotted line
    colors = ['#bb7f7e', '#e4c286', '#86b573', '#a6c2f1', '#4f7b99']  # Colors slightly darker

    # Plotting each algorithm's response time with different line styles
    for idx, (label, response_time) in enumerate(response_times_1.items()):
        ax.plot(execution_ratios, response_time, color=colors[idx], linestyle=line_styles[0], marker='o', markersize=6)

    for idx, (label, response_time) in enumerate(response_times_2.items()):
        ax.plot(execution_ratios, response_time, color=colors[idx], linestyle=line_styles[1], marker='s', markersize=6)

    for idx, (label, response_time) in enumerate(response_times_3.items()):
        ax.plot(execution_ratios, response_time, color=colors[idx], linestyle=line_styles[2], marker='^', markersize=6)

    # Adding labels and title
    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Average_Response_Time')
    ax.set_title('Average_Response_Time of Different Priorities')

    # Adjusting X and Y axis
    ax.set_xticks(execution_ratios)  # Make sure ticks are placed correctly on the x-axis
    ax.set_yticks(np.arange(7, 17, 3))  # Adjust Y-axis tick marks

    # Set Y-axis range
    ax.set_ylim(7, 17)

    # Create custom legend for line styles (Set 1, Set 2, Set 3)
    line_styles_legend = [
        plt.Line2D([0], [0], color='black', linestyle='-', label='Priority 1'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Priority 2'),
        plt.Line2D([0], [0], color='black', linestyle=':', label='Priority 3')
    ]

    # Creating a custom legend for the algorithms and data sets
    algorithm_legend = [
        plt.Line2D([0], [0], color=colors[0], linestyle='-', label='Gurobi'),
        plt.Line2D([0], [0], color=colors[1], linestyle='-', label='GSAFO'),
        plt.Line2D([0], [0], color=colors[2], linestyle='-', label='GA'),
        plt.Line2D([0], [0], color=colors[3], linestyle='-', label='Greedy'),
        plt.Line2D([0], [0], color=colors[4], linestyle='-', label='Random')
    ]

    # First legend: for algorithms (with colors)
    leg1 = ax.legend(handles=algorithm_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

    # Add the first legend manually to the current Axes
    ax.add_artist(leg1)

    # Second legend: for line styles (Set 1, Set 2, Set 3)
    ax.legend(handles=line_styles_legend, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    # Tight layout to avoid clipping
    plt.tight_layout()

    # Saving the image to the specified folder
    image_path = os.path.join(output_folder, 'Response-time-compair.png')
    plt.savefig(image_path, transparent=False)
    plt.show()