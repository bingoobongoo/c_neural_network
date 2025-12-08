import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
# 1. File Mapping (Network -> Filename)
files = {
    'A': 'small_test_scores.csv',
    'B': 'medium_test_scores.csv',
    'C': 'large_test_scores.csv'
}

# 2. Network Labels (for X-axis)
labels = {
    'A': 'Small DNN\n(953k)',
    'B': 'Medium DNN\n(3.75M)',
    'C': 'Large DNN\n(7.0M)'
}

# 3. Colors
colors = {
    'c': '#f1a226',      # Orange (My Framework)
    'keras': '#298c8c'   # Teal (Keras)
}

# 4. Legend Labels
legend_names = {
    'c': 'My Framework',
    'keras': 'Keras'
}

base_dir = "." 

def load_test_scores(filename):
    """
    Reads the test scores CSV.
    Expected format: c_score, keras_score
    Returns means and stds for both columns.
    """
    path = os.path.join(base_dir, filename)
    
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return (0, 0), (0, 0) # (mean_c, std_c), (mean_k, std_k)

    try:
        df = pd.read_csv(path)
        
        # Clean up column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Calculate stats for C
        c_mean = df['c_score'].mean()
        c_std = df['c_score'].std()
        
        # Calculate stats for Keras
        k_mean = df['keras_score'].mean()
        k_std = df['keras_score'].std()
        
        return (c_mean, c_std), (k_mean, k_std)
        
    except Exception as e:
        print(f"[ERROR] Reading {filename}: {e}")
        return (0, 0), (0, 0)

def plot_accuracy_comparison():
    print("Processing Test Scores...")
    
    # Data containers
    network_ids = ['A', 'B', 'C']
    c_means = []
    c_stds = []
    k_means = []
    k_stds = []
    
    # Load data for each network
    for net_id in network_ids:
        filename = files[net_id]
        (cm, cs), (km, ks) = load_test_scores(filename)
        c_means.append(cm)
        c_stds.append(cs)
        k_means.append(km)
        k_stds.append(ks)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.arange(len(network_ids))  # label locations
    width = 0.35  # width of the bars

    # Create bars
    rects1 = ax.bar(x - width/2, c_means, width, yerr=c_stds, label=legend_names['c'],
                    color=colors['c'], capsize=5, ecolor='#333333')
    rects2 = ax.bar(x + width/2, k_means, width, yerr=k_stds, label=legend_names['keras'],
                    color=colors['keras'], capsize=5, ecolor='#333333')

    # --- Formatting ---
    ax.set_ylabel('Test Accuracy (0.0 - 1.0)', fontsize=12)
    ax.set_title('Test Set Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([labels[id] for id in network_ids], fontsize=11)
    ax.set_ylim(0, 1.05) # Accuracy is 0-1
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # LEGEND TOP LEFT
    ax.legend(fontsize=11, loc='upper left')

    # --- Smart Label Placement ---
    def autolabel(rects, is_first_bar):
        """Attach a text label inside each bar displaying its height."""
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if height == 0: continue
            
            # Label format: Just the number (e.g., 0.515)
            # No +- sign, no std dev text
            label_text = f"{height:.3f}"
            
            # Logic: Place INSIDE at center
            xy = (rect.get_x() + rect.get_width() / 2, height / 2)
            
            # Orange gets black text, Teal gets white text
            text_color = 'black' if is_first_bar else 'white'
            
            ax.annotate(label_text,
                        xy=xy,
                        ha='center', va='center',
                        color=text_color, fontweight='bold', fontsize=10)

    autolabel(rects1, True)
    autolabel(rects2, False)

    plt.tight_layout()
    filename = "test_accuracy_comparison.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved graph to {filename}")
    plt.show()

if __name__ == "__main__":
    plot_accuracy_comparison()