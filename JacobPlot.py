import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from statsmodels.nonparametric.kde import KDEUnivariate

class RidgePlot3D:
    def __init__(self, log_dir='./logs', bandwidth=0.2, ridge_color='skyblue', fill_color='yellow'):
        self.log_dir=log_dir
        self.file_path = os.path.join(log_dir, 'jacob_storage.npy')
        self.bandwidth = bandwidth
        self.ridge_color = ridge_color
        self.fill_color = fill_color

        # Load and preprocess data
        self.jacob_data = np.load(self.file_path)
        self.transposed_data = np.transpose(self.jacob_data, (2, 0, 1, 3, 4))
        self.concatenated_data = np.concatenate(self.transposed_data, axis=-1)
        self.data_array = np.transpose(self.concatenated_data, (1, 2, 0, 3))

    def save_ridges_plot(self):
        # Set up the figure
        output_path=os.path.join(self.log_dir, 'ridge_plot.png')
        fig, axes = plt.subplots(12, 12, figsize=(20, 20), subplot_kw={'projection': '3d'})

        # Iterate over layers and heads in reverse layer order
        for layer in range(12):
            for head in range(12):
                # Select the 3 categories for the current layer and head
                subset = self.data_array[layer, head, :, :]  # Shape (3, 591)

                # Access the subplot in reversed layer order
                ax = axes[11 - layer, head]

                # Plot each category as a separate ridge
                for i in range(200):
                    category_data = subset[i, :]

                    # Compute KDE
                    x = np.linspace(0, 0.5, 100)
                    kde = KDEUnivariate(category_data)
                    kde.fit(bw=self.bandwidth)
                    y = kde.evaluate(x)

                    # Create vertices for the filled polygon
                    verts = [(x[j], i, y[j]) for j in range(len(x))] + [(x[-1], i, 0), (x[0], i, 0)]
                    poly = Poly3DCollection([verts], color=self.ridge_color, alpha=0.4)

                    # Add filled polygon to the plot
                    ax.add_collection3d(poly)

                    # Plot the line on top of the fill for clearer ridge lines
                    ax.plot(x, [i] * len(x), y, color=self.fill_color, lw=2)

                    # Adjust view to make ridges face forward
                    ax.view_init(elev=30, azim=90)
                ax.set_xlim(197, 1)

                # Remove grid lines, ticks, and customize title
                ax.grid(False)
                ax.set_facecolor("none")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_title(f"Layer {layer+1}, Head {head+1}", fontsize=6)

        # Adjust layout and save the figure
        plt.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

# Usage example:
# plotter = RidgePlot3D()
# plotter.save_ridges_plot('ridge_plot.png')
