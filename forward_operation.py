# forward_operation.py
import os
import numpy as np
import torch
from collections import defaultdict
from transformers import ViTForImageClassification
from vit_attention_analyzer import ViTAttentionAnalyzer  # Ensure this module is available
from k_value_total_plot import AttentionHeadsPlotter  # Import for any necessary plotting


def get_samples_by_class(prepared_ds, testsize):
    """Extracts a specified number of samples per class for testing."""
    class_samples = defaultdict(list)
    for i in range(len(prepared_ds["train"])):
        sample = prepared_ds["train"][i]
        label = sample['labels']
        if len(class_samples[label]) < testsize:
            class_samples[label].append(sample['pixel_values'])
        if all(len(samples) >= testsize for samples in class_samples.values()):
            break
    pixel_value_samples = torch.cat(
        [torch.stack(class_samples[label]) for label in class_samples], dim=0
    )
    return pixel_value_samples


class ForwardOperation:
    def __init__(self, model_name_or_path, prepared_ds, log_dir="./oneforward", testsize=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(model_name_or_path).to(self.device)
        self.prepared_ds = prepared_ds
        self.log_dir = log_dir
        self.testsize = testsize
        self.pixel_value_samples = get_samples_by_class(self.prepared_ds, self.testsize).to(self.device)

        self.analyzer = ViTAttentionAnalyzer(self.model)
        self.total_k_values_array = np.zeros((self.testsize, 12, 12, 197), dtype=np.float32)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def run_forward_pass(self):
        """Executes the forward pass and performs callback analysis."""
        self.model.eval()
        with torch.no_grad():
            # Perform the forward pass and get analysis data
            outputs = self.model(pixel_values=self.pixel_value_samples)
            #print(self.pixel_value_samples.shape)
            k_values_array, _ = self.analyzer.get_attention_weights_vit(self.pixel_value_samples)
'''

            if k_values_array is not None:
                # Update cumulative k-values distribution
                self.total_k_values_array = self.update_distribution(
                    self.total_k_values_array, np.transpose(np.array(k_values_array), (1, 0, 2, 3))
                )
                # Save cumulative k-values only
                self.save_totals()
                #self.generate_plots()

    def update_distribution(self, cumulative_distribution, new_array):
        """Updates cumulative distribution for k-values."""
        for sample in range(new_array.shape[0]):
            for layer in range(new_array.shape[1]):
                for head in range(new_array.shape[2]):
                    values, counts = np.unique(new_array[sample, layer, head], return_counts=True)
                    for value, count in zip(values, counts):
                        cumulative_distribution[sample, layer, head, value - 1] += count
        return cumulative_distribution

    def save_totals(self):
        """Saves cumulative k-values array only."""
        total_k_values_file_path = os.path.join(self.log_dir, 'total_k_values_array_one.npy')
        np.save(total_k_values_file_path, self.total_k_values_array)

    #def generate_plots(self):
     #   """Generates and saves plots for k-values analysis."""
      #  AttentionHeadsPlotter(self.log_dir).save_attention_heads_plot()
'''