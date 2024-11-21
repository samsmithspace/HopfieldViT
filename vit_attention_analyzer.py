import torch
import cv2
import numpy as np
import os
from list_and_tensor_operations import ListAndTensorOperations
import attentionprobvis
import plot_attention_map
class ViTAttentionAnalyzer:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device  # Get the device of the model
        self.operations = ListAndTensorOperations()


    def get_attention_weights_vit(self, pixel_values, save_dir="./attention_maps"):
        pixel_values = pixel_values.to(self.device)
        ks = []
        Jacob = []
        num_layers = len(self.model.vit.encoder.layer)
        query_outputs = [None] * num_layers
        key_outputs = [None] * num_layers

        # Define hook functions for query and key outputs
        def get_hook(layer_idx, storage):
            def hook(module, input, output):
                storage[layer_idx] = output.detach()
            return hook

        # Register hooks for all layers
        handles = []
        for i in range(num_layers):
            query_layer = self.model.vit.encoder.layer[i].attention.attention.query
            key_layer = self.model.vit.encoder.layer[i].attention.attention.key
            handles.append(query_layer.register_forward_hook(get_hook(i, query_outputs)))
            handles.append(key_layer.register_forward_hook(get_hook(i, key_outputs)))

        # Ensure pixel_values have exactly 4 dimensions (batch_size, channels, height, width)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.dim() > 4:
            pixel_values = pixel_values.squeeze()  # Remove extra singleton dimensions

        with torch.no_grad():
            self.model(pixel_values)

        # Remove the hooks
        for handle in handles:
            handle.remove()

        # Process outputs
        for i in range(num_layers):
            query = query_outputs[i]
            key = key_outputs[i]
            if query is None or key is None:
                continue

            batch_size, num_tokens, dim = query.shape
            num_heads = self.model.config.num_attention_heads
            head_dim = dim // num_heads

            # Reshape query and key to separate heads
            query = query.view(batch_size, num_tokens, num_heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, num_tokens, num_heads, head_dim).transpose(1, 2)

            # Compute attention scores and probabilities
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            #print(attention_probs.shape)
            beta = 1 / (head_dim ** 0.5)
            fro_norms = self.operations.compute_jacobian_frobenius_norm(attention_probs)

            # Compute k-values
            sorted_probs, _ = torch.sort(attention_probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            #for x in range(12):
            #    attentionprobvis.plot_heatmap(attention_probs[0, x].cpu(),i,x)


            np.save('lastlayeraverage.npy', attention_probs[0].cpu())

            #if i == 11:


            exceeds_threshold = cumulative_probs > 0.9
            exceeds_any = exceeds_threshold.any(dim=-1)
            k_indices = torch.argmax(exceeds_threshold.float(), dim=-1)
            k_indices = torch.where(exceeds_any, k_indices, attention_probs.size(-1) - 1)

            k_values = k_indices + 1  # Shape: [B, H, T]

            ks.append(k_values.cpu().tolist())
            Jacob.append(fro_norms.cpu().tolist())

            '''
            plot_attention_map.visualize_attention(
                attentions=attention_probs,
                img_path='./sample1.jpg',
                patch_size=16,
                output_dir="visualizations",
                threshold=0.8,
                i=i
            )
            '''

        return ks, Jacob
