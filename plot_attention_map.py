import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def visualize_attention(
        attentions,
        img_path,
        patch_size=16,
        output_dir="visualizations",
        threshold=None,
        i=0
):
    """
    Visualize self-attention maps.

    Parameters:
        attentions (torch.Tensor): Attention tensor of shape (batch_size, num_heads, num_queries, num_keys).
        img_path (str): Path to the local image file.
        patch_size (int): Patch size of the model.
        output_dir (str): Base directory name for visualizations.
        threshold (float): Threshold for attention visualization (default: None).
        i (int): Integer to append to the directory name (default: 0).

    Returns:
        None. Saves attention maps and visualizations to `output_dir-i`.
    """
    # Append the integer `i` to the output directory
    output_dir = f"{output_dir}-{i}"
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess the image
    img = Image.open(img_path).convert('RGB')
    transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),  # Resize to match model's input size
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # Make the image divisible by patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    # Feature map dimensions
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    # Extract attention maps for CLS token
    attentions = attentions[0, :, 0, 1:].reshape(attentions.shape[1], -1)
    print(attentions.shape)
    if threshold is not None:
        # Keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(attentions.shape[0]):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(attentions.shape[0], w_featmap, h_featmap).float()
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
            0].cpu().numpy()

    attentions = attentions.reshape(attentions.shape[0], w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
        0].cpu().numpy()

    # Save original image
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                 os.path.join(output_dir, "img.png"))

    # Save attention maps
    for j in range(attentions.shape[0]):
        fname = os.path.join(output_dir, f"attn-head{j}.png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    # Save thresholded visualizations if threshold is provided
    if threshold is not None:
        image = skimage.io.imread(os.path.join(output_dir, "img.png"))
        for j in range(attentions.shape[0]):
            display_instances(
                image,
                th_attn[j],
                fname=os.path.join(output_dir, f"mask_th{threshold}_head{j}.png"),
                blur=False
            )