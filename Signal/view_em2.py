import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
import os

def main():
    # Find all npz files in the current directory
    npz_files = glob.glob("*.npz")
    
    if not npz_files:
        print("No .npz files found in the current directory.")
        sys.exit(1)
        
    print("Found the following .npz files:")
    for i, f in enumerate(npz_files):
        print(f"[{i}] {f}")
        
    target_npz = npz_files[0]
    target_csv = target_npz.replace(".npz", ".csv")
    print(f"\nLoading {target_npz} by default...")
    
    try:
        data = np.load(target_npz)
        if "X_em2_tensors" not in data or "event_nums" not in data:
            print(f"Error: Required keys not found in {target_npz}.")
            sys.exit(1)
            
        em2_raw = data["X_em2_tensors"]
        ev_nums = data["event_nums"]
        
        # Flatten the event and TOB object dimensions together
        em2_matrices = em2_raw.reshape(-1, 12, 12)
        N = em2_matrices.shape[0]
        print(f"Successfully loaded and reshaped to {N} individual 12x12 matrices.")
        
    except Exception as e:
        print(f"Failed to load NPZ data: {e}")
        sys.exit(1)

    # Load corresponding CSV and create a fast lookup dictionary
    signal_lookup = {}
    if os.path.exists(target_csv):
        print(f"Loading corresponding CSV: {target_csv}")
        try:
            df = pd.read_csv(target_csv)
            # Create a dictionary mapping (eventNumber, tob_index) -> signal
            # This perfectly mimics your pipeline's alignment logic but is much faster for a visualizer
            signal_lookup = df.set_index(['eventNumber', 'tob_index'])['signal'].to_dict()
        except Exception as e:
            print(f"Warning: Failed to parse CSV data: {e}")
    else:
        print(f"Warning: Corresponding CSV ({target_csv}) not found. Signal labels won't be shown.")

    # Visualization Setup
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title(f"EM2 Viewer - {target_npz}")
    
    current_idx = [0] 

    def draw_matrix(idx):
        ax.clear()
        matrix = em2_matrices[idx]
        
        ax.imshow(matrix, cmap='gray', interpolation='nearest')
        
        val_min, val_max = np.min(matrix), np.max(matrix)
        if val_min == val_max:
            threshold = val_min
        else:
            threshold = val_min + (val_max - val_min) / 2.0
        
        for i in range(12):
            for j in range(12):
                val = matrix[i, j]
                text_color = 'black' if val > threshold else 'white'
                
                text_str = f"{val:.2f}" if abs(val) > 0.005 else "0"
                ax.text(j, i, text_str, ha='center', va='center', color=text_color, fontsize=9)
                
        # Calculate event alignment exactly like the training pipeline
        event_idx = idx // 6
        tob_idx = idx % 6
        actual_event_num = ev_nums[event_idx]
        
        # Determine Signal/Background status
        lookup_key = (actual_event_num, tob_idx)
        if lookup_key in signal_lookup:
            signal_val = signal_lookup[lookup_key]
            signal_text = "SIGNAL (Tau)" if signal_val == 1 else "BACKGROUND"
            title_color = "green" if signal_val == 1 else "red"
        else:
            signal_text = "UNKNOWN (Not in CSV)"
            title_color = "black"

        ax.set_title(
            f"Matrix {idx} / {N - 1} | {signal_text}\n(Event Num: {actual_event_num}, TOB: {tob_idx})", 
            fontsize=14, color=title_color, fontweight='bold'
        )
        
        ax.set_xticks(np.arange(12))
        ax.set_yticks(np.arange(12))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.tick_params(which='minor', size=0)
        
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            current_idx[0] = (current_idx[0] + 1) % N
            draw_matrix(current_idx[0])
        elif event.key == 'left':
            current_idx[0] = (current_idx[0] - 1) % N
            draw_matrix(current_idx[0])

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    draw_matrix(current_idx[0])
    
    print("\nViewer launched! Click on the plot window and use the Left/Right arrow keys to scroll.")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()