import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# --- Configuration ---
# 1. Network Definitions
networks = {
    'A': 'DNN\n(1.47M params)',
    'B': 'CNN\n(79k params)'
}

# 2. Configurations (Bars)
configs = {
    '1': 'No Inline',
    '2': 'Inline Functions'
}

# 3. Colors
colors = {
    '1': '#f1a226',  # Orange
    '2': '#298c8c'   # Teal
}

base_dir = "." 

def load_network_data(config_num, network_id):
    """
    Loads data for a specific config and network.
    Returns the mean time and standard deviation.
    """
    folder_name = f"results_{config_num}_{network_id}"
    folder_path = os.path.join(base_dir, folder_name)
    all_times = []
    
    if not os.path.exists(folder_path):
        return 0, 0

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            times = df['time'].tolist()
            all_times.extend(times)
        except Exception as e:
            print(f"   [ERROR] Reading {file}: {e}")

    if not all_times:
        return 0, 0
        
    return np.mean(all_times), np.std(all_times)

def plot_inline_comparison():
    print("Processing Data...")
    
    # Prepare data structures for plotting
    network_ids = list(networks.keys())
    network_labels = list(networks.values())
    
    means = {cid: [] for cid in configs}
    stds = {cid: [] for cid in configs}
    
    # Load data
    for net_id in network_ids:
        for config_id in configs:
            m, s = load_network_data(config_id, net_id)
            means[config_id].append(m)
            stds[config_id].append(s)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.arange(len(network_labels))  # label locations
    width = 0.35  # width of the bars

    # Create bars for each configuration
    rects1 = ax.bar(x - width/2, means['1'], width, yerr=stds['1'], label=configs['1'],
                    color=colors['1'], capsize=5, ecolor='#333333')
    rects2 = ax.bar(x + width/2, means['2'], width, yerr=stds['2'], label=configs['2'],
                    color=colors['2'], capsize=5, ecolor='#333333')

    # --- Formatting ---
    ax.set_ylabel('Average Epoch Time (seconds)', fontsize=12)
    ax.set_title('Impact of Inline Functions on Training Speed', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(network_labels, fontsize=11)
    ax.legend(fontsize=11)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # --- Smart Label Placement ---
    def autolabel(rects, config_id):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if height == 0: continue
            
            # Find the std dev for this specific bar
            std_val = stds[config_id][i]

            label_text = f"{height:.2f} $\pm$ {std_val:.2f}"
            
            # Logic: Inside if tall enough, Above if short
            # Use a threshold relative to the max height in the whole plot
            max_h = max(max(means['1']), max(means['2'])) if means['1'] or means['2'] else 1
            
            if height < (0.15 * max_h):
                # Place above
                xy = (rect.get_x() + rect.get_width() / 2, height + (0.02 * max_h))
                text_color = 'black'
                va = 'bottom'
            else:
                # Place inside
                xy = (rect.get_x() + rect.get_width() / 2, height / 2)
                text_color = 'black' if config_id == '1' else 'white'
                va = 'center'

            ax.annotate(label_text,
                        xy=xy,
                        ha='center', va=va,
                        color=text_color, fontweight='bold', fontsize=10)

    autolabel(rects1, '1')
    autolabel(rects2, '2')

    plt.tight_layout()
    filename = "inline_functions_impact.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved graph to {filename}")
    plt.show()

if __name__ == "__main__":
    plot_inline_comparison()