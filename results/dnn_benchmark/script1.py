import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# --- Configuration ---
# 1. Networks
networks = {
    'A': 'DNN (953k params)',
    'B': 'DNN (3.75M params)',
    'C': 'DNN (7.0M params)'
}

# 2. Configurations
configs = {
    '1': 'My Framework',
    '2': 'Keras'
}

# 3. Visuals
colors = {
    '1': '#f1a226',  # Orange (My Framework)
    '2': '#298c8c'   # Teal (Keras)
}

base_dir = "."

def get_aggregated_data(config_id, net_id):
    """
    Reads all CSV files for a config/network.
    Returns:
      - mean_losses: Series of average loss per epoch (index=epoch)
      - avg_time: scalar average time per epoch
      - std_time: scalar standard deviation of time
    """
    folder = f"results_{config_id}_{net_id}"
    path = os.path.join(base_dir, folder)
    
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not all_files:
        print(f"[WARNING] No files found for {folder}")
        return None, 0, 0

    # Read all CSVs into a single DataFrame
    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return None, 0, 0
    
    combined = pd.concat(dfs)
    
    # 1. Calculate Mean Loss per Epoch (for Line Graph)
    # Group by epoch and take the mean of 'train_loss' across all runs
    loss_by_epoch = combined.groupby('epoch')['train_loss'].mean()
    
    # 2. Calculate Time Stats (for Bar Graph)
    # We take all time values from all epochs/runs
    all_times = combined['time']
    avg_time = all_times.mean()
    std_time = all_times.std()
    
    return loss_by_epoch, avg_time, std_time

def create_network_figure(net_id, net_name):
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Data Gathering ---
    data = {}
    for cid in configs:
        data[cid] = get_aggregated_data(cid, net_id)

    # --- Graph 1: Training Loss (Line Plot) ---
    for cid, label in configs.items():
        loss_series, _, _ = data[cid]
        if loss_series is not None:
            ax1.plot(loss_series.index, loss_series.values, 
                     label=label, color=colors[cid], linewidth=2.5, marker='o', markersize=4)
    
    ax1.set_title(f'Training Loss Convergence', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)

    # --- Graph 2: Average Epoch Time (Bar Plot) ---
    bar_labels = []
    bar_means = []
    bar_stds = []
    bar_colors = []
    
    for cid, label in configs.items():
        _, mean_t, std_t = data[cid]
        bar_labels.append(label)
        bar_means.append(mean_t)
        bar_stds.append(std_t)
        bar_colors.append(colors[cid])
        
    x_pos = np.arange(len(bar_labels))
    bars = ax2.bar(x_pos, bar_means, yerr=bar_stds, align='center', 
                   color=bar_colors, ecolor='#333333', capsize=10, width=0.5)
    
    ax2.set_title(f'Average Training Speed', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time per Epoch (seconds)', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bar_labels, fontsize=11)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)

    # Annotate bars
    for i, rect in enumerate(bars):
        height = rect.get_height()
        label_text = f"{height:.2f}s"
        # Place label inside or above based on height
        xy = (rect.get_x() + rect.get_width() / 2, height / 2)
        ax2.annotate(label_text, xy=xy, ha='center', va='center', 
                     color='white', fontweight='bold', fontsize=11)

    # --- Final Figure Formatting ---
    fig.suptitle(f'Benchmark Results: Network {net_id}\n{net_name}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    filename = f"benchmark_network_{net_id}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved figure: {filename}")
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    for net_id, net_name in networks.items():
        print(f"Processing {net_id}...")
        create_network_figure(net_id, net_name)