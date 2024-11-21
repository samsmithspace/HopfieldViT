import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class AttentionHeadsPlotter:
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        self.file_path = os.path.join(log_dir, 'total_k_values_array.npy')
        # Load and preprocess data
        self.kvalue_data = np.load(self.file_path)
        self.summed_array = np.sum(self.kvalue_data, axis=0)

    def save_attention_heads_plot(self):
        save_path = os.path.join(self.log_dir, 'attention_heads_all_layers_total.png')
        """
        Plots the distribution of k-value counts for each attention head in all layers as violin plots (with swapped axes)
        and saves the plot locally. Each row represents a layer, and each subplot represents a head. Displays the median
        k-value index on each subplot.

        Parameters:
        - save_path: The file path where the plot image will be saved.
        """
        # Ensure the array is in integer format
        k_values_list = self.summed_array.astype(int)

        num_layers = k_values_list.shape[0]
        num_heads = k_values_list.shape[1]
        num_tokens = k_values_list.shape[2]

        # Thresholds for color backgrounds
        threshold1 = num_tokens / 2
        threshold2 = num_tokens / 8
        threshold3 = num_tokens / 32

        # Colors for background based on median k-value index
        color1 = '#FFCCCC'  # Light red
        color2 = '#FFFFCC'  # Light yellow
        color3 = '#CCFFCC'  # Light green
        color4 = '#CCCCFF'  # Light blue

        # Create a grid of subplots: 12 rows (layers), 12 columns (heads)
        fig, axes = plt.subplots(nrows=num_layers, ncols=num_heads, figsize=(24, 24))
        fig.suptitle("k-value Count Distribution for Attention Heads Across 12 Layers", fontsize=20)

        # Flatten the axes array in case it's returned as 1D
        axes = axes.flatten()

        # Plot the distribution of k-value counts for each head
        for layer in range(num_layers - 1, -1, -1):  # Iterate over layers in reverse
            for head in range(num_heads):
                # Reverse the row index to match Layer 12 at the top, Layer 1 at the bottom
                ax = axes[
                    (num_layers - 1 - layer) * num_heads + head]  # Access the correct subplot in the flattened array

                # Get the count distribution for the current layer and head
                k_value_counts = k_values_list[layer, head]

                # Repeat each index according to its count to create a distribution-like array
                distribution = np.repeat(np.arange(1, num_tokens + 1), k_value_counts)

                # Calculate the median k-value index
                k_median = np.median(distribution)

                # Assign background color based on the median index
                if k_median > threshold1:
                    ax.set_facecolor(color1)
                elif threshold2 < k_median <= threshold1:
                    ax.set_facecolor(color2)
                elif threshold3 < k_median <= threshold2:
                    ax.set_facecolor(color3)
                else:
                    ax.set_facecolor(color4)

                # Create the violin plot for the current head with swapped axes (horizontal orientation)
                sns.violinplot(ax=ax, x=distribution, color='blue', inner='quartile', orient='h')
                ax.set_title(f"L{layer + 1}H{head + 1}")
                ax.set_xlabel("k-value")
                ax.set_ylabel("Density")

                # Display the median index on the plot
                ax.text(0.5, 0.9, f"Median: {k_median:.2f}", transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="black", bbox=dict(facecolor='white', alpha=0.6))

                # Set the x-axis range to be fixed between 1 and 197
                ax.set_xlim(1, num_tokens)

        # Adjust layout to avoid overlap
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure to the specified path
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")

# Usage example:
# plotter = AttentionHeadsPlotter()
# plotter.save_attention_heads_plot("attention_heads_all_layers_total.png")
