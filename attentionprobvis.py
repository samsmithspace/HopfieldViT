import os

import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(input_array, layer, head, title="Heatmap", xlabel="X-axis", ylabel="Y-axis", color_map="viridis",folder="heatmaps"):
    """
    Plots a heatmap and saves it as a local image.

    Parameters:
    - input_array: numpy array of shape (197, 197), the data for the heatmap.
    - layer: int, the layer number to distinguish the file name.
    - head: int, the head number to distinguish the file name.
    - title: str, the title of the heatmap (default: "Heatmap").
    - xlabel: str, the label for the X-axis (default: "X-axis").
    - ylabel: str, the label for the Y-axis (default: "Y-axis").
    - color_map: str, the colormap to use for the heatmap (default: "viridis").
    """
    if input_array.shape != (197, 197):
        raise ValueError("Input array must have dimensions 197x197.")
    os.makedirs(folder, exist_ok=True)
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(input_array, cmap=color_map, aspect="auto")
    plt.tight_layout()

    # Save the plot as a local image
    filename = os.path.join(folder, f"heatmap_layer{layer}_head{head}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Heatmap saved as {filename}")
