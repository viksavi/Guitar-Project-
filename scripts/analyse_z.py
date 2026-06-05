import json
import math
import matplotlib.pyplot as plt
import numpy as np

def compute_avg(d):
    N = len(d)
    sum_z = 0

    for i in range(1, N):
        i = str(i)      # key is a string
        z = d[i][0][2]  # z value
        sum_z += z

    return sum_z / N

def compute_variance(d, avg):
    N = len(d)
    sum_z = 0

    for i in range(1, N):
        i = str(i)      # key is a string
        z = d[i][0][2]  # z value
        sum_z += (z - avg) * (z - avg)

    return sum_z / N

def compute_min_max(d):
    N = len(d)
    min_z = 10
    max_z = 0

    for i in range(1, N):
        i = str(i)      # key is a string
        z = d[i][0][2]  # z value
        if z < min_z:
            min_z = z
        if z > max_z:
            max_z = z

    return (min_z, max_z)

def plot_data(d, mean_val, std_dev, min_val, max_val):
    np.random.seed(42)
    N = len(d)
    raw_data = np.array([d[str(i)][0][2]*100 for i in range(1, N)])

    # --- 2. CALCULATE SUMMARY STATISTICS ---
    mean_val = np.mean(raw_data)
    std_dev = np.std(raw_data)
    min_val = np.min(raw_data)
    max_val = np.max(raw_data)

    print("average z depth value:", mean_val)
    print("standard deviation of z depth value:", std_dev)
    print("min/max z depth value:", min_val, ":", max_val)

    # Calculate asymmetric error lengths required by matplotlib:
    # Range error: [[mean - min], [max - mean]]
    min_max_errors = [[mean_val - min_val], [max_val - mean_val]]
    # SD error: [[std], [std]]
    std_errors = [[std_dev], [std_dev]]

    # --- 3. GENERATE THE PLOT ---
    plt.figure(figsize=(5, 6))
    x_position = 1

    # LAYER 1 (Background): The Raw Data
    # Plot as a faint cloud with high transparency (alpha=0.15)
    plt.scatter(x=np.full_like(raw_data, x_position-0.05), y=raw_data, 
                color='lightblue', alpha=0.1, s=35, label='Z values (cm)')

    # LAYER 2 (Mid-ground): The Outer Range (Min/Max)
    # Simple thin gray error bar representing the extreme bounds
    plt.errorbar(x=x_position, y=mean_val, yerr=min_max_errors, fmt='none', 
                 ecolor='gray', elinewidth=1.5, capsize=20, label='Min/Max Range')

    # LAYER 3 (Foreground): The Central Metrics (Mean/SD)
    # Thick blue inner error bar with a central mean marker
    plt.errorbar(x=x_position, y=mean_val, yerr=std_errors, fmt='o', 
                 color='blue', elinewidth=4, capsize=8, markersize=8, label='Mean $\pm$ Std Dev')

    # --- 4. CLEAN UP & DISPLAY ---
    plt.xlim(0.5, 1.5)
    plt.xticks([x_position], ['video_a_p_s'])
    plt.ylabel('Values')
    plt.xlabel('centimeters')
    plt.title('Guitar depth (Z) value estimation from ArUco markers')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(axis='y', linestyle=':', alpha=0.6)

    plt.show()

def compute_stats(d):
    avg_z = compute_avg(d)
    var_z = compute_variance(d, avg_z)
    min_z, max_z = compute_min_max(d)
    std_z = math.sqrt(var_z)

    plot_data(d, avg_z, std_z, min_z, max_z)

def main():
    path = "./data/video_a_p_s/guitar/outdata.json"

    with open(path) as f:
        d = json.load(f)

    compute_stats(d)

main()
