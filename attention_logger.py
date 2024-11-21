import os
import numpy as np
import torch
from collections import defaultdict
from transformers import TrainerCallback
from vit_attention_analyzer import ViTAttentionAnalyzer  # Ensure this module is available
from list_and_tensor_operations import ListAndTensorOperations
from JacobPlot import RidgePlot3D
from k_value_change_plot import KValueRidgePlot3D
from k_value_total_plot import AttentionHeadsPlotter
import sys
import time
import random


def get_samples_by_class(prepared_ds, testsize):
    start_time = time.time()
    class_samples = defaultdict(list)

    for sample in prepared_ds["train"]:
        label = sample['labels']
        class_samples[label].append(sample['pixel_values'])

    selected_samples = []
    for label in class_samples:
        if len(class_samples[label]) >= testsize:
            selected_samples.extend(random.sample(class_samples[label], testsize))
        else:
            selected_samples.extend(class_samples[label])

    pixel_value_samples = torch.stack(selected_samples)
    total_sample_size = len(selected_samples)

    print(f"Time taken for get_samples_by_class: {time.time() - start_time:.4f} seconds")
    return pixel_value_samples, total_sample_size


class DistributionUpdater:
    @staticmethod
    def update_distribution(cumulative_distribution, new_array):
        start_time = time.time()
        for sample in range(new_array.shape[0]):
            for layer in range(new_array.shape[1]):
                for head in range(new_array.shape[2]):
                    values, counts = np.unique(new_array[sample, layer, head], return_counts=True)
                    for value, count in zip(values, counts):
                        cumulative_distribution[sample, layer, head, value - 1] += count
        print(f"Time taken for update_distribution: {time.time() - start_time:.4f} seconds")
        return cumulative_distribution


class AttentionLoggerCallback(TrainerCallback):
    def __init__(self, model, prepared_ds, log_dir="./logs", max_steps=50, save_interval=1):
        self.model = model
        self.prepared_ds = prepared_ds
        self.log_dir = log_dir
        self.max_steps = max_steps
        self.testsizeini = 1

        start_time = time.time()
        self.pixel_value_samples, self.testsize = get_samples_by_class(self.prepared_ds, self.testsizeini)
        print(f"Initialization: testsize = {self.testsize}")
        print(f"Time taken for initial sample extraction: {time.time() - start_time:.4f} seconds")

        self.save_interval = save_interval
        self.total_k_values_array = np.zeros((self.testsize, 12, 12, 197), dtype=np.float32)
        self.k_values_storage = np.zeros((max_steps, 12, self.testsize, 12, 197), dtype=np.float32)
        self.total_attention_probs_array = np.zeros((12, self.testsize, 12, 197, 197), dtype=np.float32)
        self.jacob_storage = np.zeros((max_steps, 12, self.testsize, 12), dtype=np.float32)
        self.current_step = 0

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.analyzer = ViTAttentionAnalyzer(self.model)

    def check_memory_usage(self):
        memory_usage = {
            'total_k_values_array': self.total_k_values_array.nbytes,
            'k_values_storage': self.k_values_storage.nbytes,
            'total_attention_probs_array': self.total_attention_probs_array.nbytes,
            'jacob_storage': self.jacob_storage.nbytes,
            'analyzer': sys.getsizeof(self.analyzer)
        }
        print("Memory usage (in bytes):", memory_usage)

    def on_step_end(self, args, state, control, **kwargs):
        if self.current_step < self.max_steps:
            print(f"Step {self.current_step} of {self.max_steps}")

            step_start_time = time.time()

            k_values_array, Jacob = self.analyzer.get_attention_weights_vit(self.pixel_value_samples)

            print(f"Time for get_attention_weights_vit: {time.time() - step_start_time:.4f} seconds")

            if k_values_array is not None and Jacob is not None:
                storage_time_start = time.time()
                self.k_values_storage[self.current_step] = k_values_array
                self.jacob_storage[self.current_step] = Jacob

                transposed_data = np.transpose(np.array(k_values_array), (1, 0, 2, 3))
                self.total_k_values_array = DistributionUpdater.update_distribution(self.total_k_values_array,
                                                                                    transposed_data)
                print(
                    f"Time taken for storing data and updating distribution: {time.time() - storage_time_start:.4f} seconds")

                self.current_step += 1

                if self.current_step % self.save_interval == 0:
                    save_start_time = time.time()
                    self.save_storage()
                    print(f"Time taken for save_storage: {time.time() - save_start_time:.4f} seconds")
                    print(f"Saved storage at step {self.current_step}")

            self.check_memory_usage()

            if self.current_step == self.max_steps:
                print("Reached max steps, saving totals.")
                total_save_start_time = time.time()
                self.save_totals()
                print(f"Time taken for save_totals: {time.time() - total_save_start_time:.4f} seconds")

                #plot_time_start = time.time()
                #RidgePlot3D().save_ridges_plot()
                #KValueRidgePlot3D().save_ridges_plot()
                #AttentionHeadsPlotter(self.log_dir).save_attention_heads_plot()
                #print(f"Time taken for generating plots: {time.time() - plot_time_start:.4f} seconds")

    def save_storage(self):
        k_values_file_path = os.path.join(self.log_dir, 'k_values_storage.npy')
        np.save(k_values_file_path, self.k_values_storage)

        jacob_file_path = os.path.join(self.log_dir, 'jacob_storage.npy')
        np.save(jacob_file_path, self.jacob_storage)

    def save_totals(self):
        total_k_values_file_path = os.path.join(self.log_dir, 'total_k_values_array.npy')
        np.save(total_k_values_file_path, self.total_k_values_array)
