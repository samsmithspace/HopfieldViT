import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from statsmodels.nonparametric.kde import KDEUnivariate

class KValueRidgePlot3D:
    def __init__(self, log_dir='./logs', bandwidth=0.3, ridge_color='skyblue', fill_color='yellow'):
        self.log_dir = log_dir
        self.file_path = os.path.join(log_dir, 'k_values_storage.npy')
        self.bandwidth = bandwidth
        self.ridge_color = ridge_color
        self.fill_color = fill_color

        # Load and preprocess data
        self.kvalue_data = np.load(self.file_path)
        self.transposed_data = np.transpose(self.kvalue_data, (2, 0, 1, 3, 4))
        self.concatenated_data = np.concatenate(self.transposed_data, axis=-1)  # Shape becomes (9, 12, 12, 591)
        self.data_array = np.transpose(self.concatenated_data, (1, 2, 0, 3))  # Shape becomes (12, 12, 3, 591)

    def save_ridges_plot(self):
        # Set up the figure
        output_path = os.path.join(self.log_dir, 'kvalue_ridge_plot.png')
        fig, axes = plt.subplots(12, 12, figsize=(20, 20), subplot_kw={'projection': '3d'})

        # Iterate over layers and heads in reverse layer order
        for layer in range(12):
            for head in range(12):
                # Select the 3 categories for the current layer and head
                subset = self.data_array[layer, head, :, :]  # Shape (3, 591)

                # Access the subplot in reversed layer order
                ax = axes[11 - layer, head]

                # Plot each category as a separate ridge
                for i in range(200):  # Assuming 3 categories
                    category_data = subset[i, :]

                    # Compute KDE
                    x = np.linspace(1, 197, 100)
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
                    ax.view_init(elev=45, azim=90)

                # Set x-axis limits in reverse to flip the axis
                ax.set_xlim(197, 1)

                # Customize subplot appearance
                ax.set_facecolor("none")
                ax.set_axis_off()
                ax.set_title(f"Layer {layer+1}, Head {head+1}", fontsize=6, pad=10)

        # Adjust layout and save the figure
        plt.tight_layout()
        fig.savefig(output_path, dpi=600)
        plt.close(fig)

# Usage example:
# plotter = KValueRidgePlot3D()
# plotter.save_ridges_plot('kvalue_ridge_plot.png')
