import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# --- Configuration ---
# 1. Colors: 
# Orange (#f1a226) for Naive
# Teal (#298c8c) for Cache Optimized
colors = ['#f1a226', '#298c8c']

# 2. Labels for the X-axis
labels = [
    'Network A\n(Naive)',
    'Network B\n(Cache Opt.)'
]

# 3. Define the specific targets to load: (Config Number, Network Layout)
# Config 1 = Naive, Config 2 = Cache Optimized
targets = [
    ('1', 'A'),  # Naive for Network A
    ('2', 'B')   # Cache Optimized for Network B
]

base_dir = "." 

def load_target_data(config_num, network_layout):
    """
    Loads all 5 CSV files for a specific config and layout.
    """
    folder_name = f"results_{config_num}_{network_layout}"
    folder_path = os.path.join(base_dir, folder_name)
    all_times = []
    
    # DEBUG: Check if folder exists
    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder not found: {folder_path}")
        return []

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"   -> Loading Config {config_num} (Net {network_layout}): Found {len(csv_files)} files.")
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Filter out warmup epoch if needed, e.g., df['time'][1:]
            times = df['time'].tolist()
            all_times.extend(times)
        except Exception as e:
            print(f"   [ERROR] Reading {file}: {e}")

    return all_times

def plot_comparison():
    means = []
    std_devs = []
    
    print("Processing Data...")
    
    # Load Data for the specific targets
    for (config_num, net_layout) in targets:
        times = load_target_data(config_num, net_layout)
        
        if not times:
            means.append(0)
            std_devs.append(0)
        else:
            means.append(np.mean(times))
            std_devs.append(np.std(times))

    # Check if we have data
    if sum(means) == 0:
        print("[ERROR] No data found for targets. Check your folder paths.")
        return

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 7))
    x_pos = np.arange(len(labels))
    
    # Draw Bars
    bars = ax.bar(x_pos, means, yerr=std_devs, align='center', 
                  color=colors, ecolor='#333333', capsize=10, width=0.5)
    
    ax.set_ylabel('Average Epoch Time (seconds)', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_title('Cross-Network Performance Comparison', fontsize=14, fontweight='bold')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # --- Smart Label Placement ---
    max_height = max(means) if means else 1
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        mean_val = means[i]
        std_val = std_devs[i]
        
        if height == 0:
            continue

        label_text = f"{mean_val:.2f} $\pm$ {std_val:.2f}"
        
        # Text placement logic
        if height < (0.15 * max_height):
            # Place above (Black) if bar is too short
            text_x = bar.get_x() + bar.get_width() / 2
            text_y = height + (0.02 * max_height)
            text_color = 'black'
            va = 'bottom'
        else:
            # Place inside (Center)
            text_x = bar.get_x() + bar.get_width() / 2
            text_y = height / 2
            # Orange gets black text, Teal gets white text
            text_color = 'black' if i == 0 else 'white'
            va = 'center'

        ax.text(text_x, text_y, label_text,
                ha='center', va=va, 
                color=text_color, fontweight='bold', fontsize=11)

    plt.tight_layout()
    filename = "comparison_A_vs_B_cache.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved comparison graph to {filename}")
    # plt.show()

if __name__ == "__main__":
    plot_comparison()