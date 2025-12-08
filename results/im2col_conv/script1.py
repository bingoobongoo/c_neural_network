import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# --- Configuration ---
# 1. Network Parameters (X-Axis)
network_params = {
    'A': 13930,
    'B': 79658,
    'C': 184074
}

# 2. Configurations (Lines)
configs = {
    '1': 'Classic Convolution',
    '2': 'Im2Col Algorithm'
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
        # Return dummy data 0,0 if folder is missing so script doesn't crash
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

def plot_scalability_graph():
    print("Processing Data...")
    
    # Sort networks by parameter count to ensure X-axis is ordered
    sorted_nets = sorted(network_params.keys(), key=lambda x: network_params[x])
    x_values = [network_params[net] for net in sorted_nets]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Loop through each configuration to draw a line
    for config_id, config_label in configs.items():
        y_means = []
        y_stds = []
        
        for net_id in sorted_nets:
            mean, std = load_network_data(config_id, net_id)
            y_means.append(mean)
            y_stds.append(std)
            
        # Plot Line with Error Bars
        ax.errorbar(x_values, y_means, yerr=y_stds, label=config_label,
                    color=colors[config_id], marker='o', capsize=5, 
                    linestyle='-', linewidth=2, markersize=4)

        # --- Conditional Annotation with Ranges ---
        for i, val in enumerate(y_means):
            if val > 0:
                std_val = y_stds[i]
                
                # Logic: One Above, One Below
                if config_id == '1': 
                    # Classic Conv -> Above
                    offset = (0, 10)
                    va = 'bottom'
                else: 
                    # Im2Col -> Below
                    offset = (0, -15) 
                    va = 'top'

                # Format: "Mean ± Std s"
                label_text = f"{val:.2f} $\pm$ {std_val:.2f}s"

                ax.annotate(label_text, 
                            (x_values[i], val), 
                            xytext=offset, textcoords='offset points', 
                            ha='center', va=va,
                            fontsize=9, color=colors[config_id], fontweight='bold')

    # --- Formatting ---
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Average Epoch Time (seconds)', fontsize=12)
    ax.set_title('Convolution Algorithm Scalability Analysis', fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Legend
    ax.legend(fontsize=11)
    
    # X-Axis Ticks
    ax.set_xticks(x_values)
    ax.set_xticklabels([f"{x:,}" for x in x_values]) 

    # Margins padding to ensure text doesn't get cut off
    ax.margins(x=0.1, y=0.1)

    plt.tight_layout()
    filename = "conv_algorithm_scalability.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved graph to {filename}")
    # plt.show()

if __name__ == "__main__":
    plot_scalability_graph()