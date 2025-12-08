import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# --- Configuration ---
# 1. Colors: Orange, Teal, Muted Red
colors = ['#f1a226', '#298c8c', '#8c2929']

# 2. Config Names
config_names = {
    '1': 'Naive\n(No Cache/BLAS)',
    '2': 'Cache\nOptimized',
    '3': 'BLAS'
}

networks = ['A', 'B']
base_dir = "." 

def load_experiment_data(config_num, network_layout):
    folder_name = f"results_{config_num}_{network_layout}"
    folder_path = os.path.join(base_dir, folder_name)
    all_times = []
    
    # DEBUG: Check if folder exists
    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder not found: {folder_path}")
        return []

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"   -> Config {config_num} (Net {network_layout}): Found {len(csv_files)} CSV files.")
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Filter out warmup epoch if needed, e.g., df['time'][1:]
            times = df['time'].tolist()
            all_times.extend(times)
        except Exception as e:
            print(f"   [ERROR] Reading {file}: {e}")

    return all_times

def plot_network_performance(network_layout):
    means = []
    std_devs = []
    labels = []
    
    # Load Data
    for config_num in ['1', '2', '3']:
        times = load_experiment_data(config_num, network_layout)
        if not times:
            means.append(0)
            std_devs.append(0)
        else:
            means.append(np.mean(times))
            std_devs.append(np.std(times))
        labels.append(config_names[config_num])

    # Check if we have any data to plot
    if sum(means) == 0:
        print(f"[ERROR] No data found for Network {network_layout}. Skipping plot generation.")
        return

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))
    x_pos = np.arange(len(labels))
    
    # Draw Bars
    bars = ax.bar(x_pos, means, yerr=std_devs, align='center', 
                  color=colors, ecolor='#333333', capsize=10, width=0.6)
    
    ax.set_ylabel('Average Epoch Time (seconds)', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(f'Training Performance: Network {network_layout}', fontsize=14, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # --- Smart Label Placement ---
    max_height = max(means) if means else 1
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        mean_val = means[i]
        std_val = std_devs[i]
        
        # If the bar is 0 (no data), skip label
        if height == 0:
            continue

        label_text = f"{mean_val:.2f} $\pm$ {std_val:.2f}"
        
        # Logic: If bar is very short (< 10% of the tallest bar), 
        # put text ABOVE the bar in BLACK so it is readable.
        # Otherwise, put it INSIDE the bar.
        if height < (0.10 * max_height):
            # Place above
            text_x = bar.get_x() + bar.get_width() / 2
            text_y = height + (0.02 * max_height) # Slightly above error bar
            text_color = 'black'
            va = 'bottom'
        else:
            # Place inside (Center)
            text_x = bar.get_x() + bar.get_width() / 2
            text_y = height / 2
            # Orange bar gets black text, others get white
            text_color = 'black' if i == 0 else 'white'
            va = 'center'

        ax.text(text_x, text_y, label_text,
                ha='center', va=va, 
                color=text_color, fontweight='bold', fontsize=11)

    plt.tight_layout()
    filename = f"performance_network_{network_layout}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved graph to {filename}")
    # plt.show() # Uncomment if running in a windowed environment

if __name__ == "__main__":
    for net in networks:
        print(f"Processing Network {net}...")
        plot_network_performance(net)