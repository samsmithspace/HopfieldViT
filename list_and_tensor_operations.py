import numpy as np
import torch


class ListAndTensorOperations:
    def __init__(self):
        pass

    def get_list_dimensions(self, lst):
        """
        Recursively gets the dimensions of a nested list.

        Args:
            lst (list): A nested list whose dimensions need to be calculated.

        Returns:
            list: A list containing the dimensions at each level.
        """
        dimensions = []
        while isinstance(lst, list):
            dimensions.append(len(lst))
            lst = lst[0] if lst else []
        return dimensions

    def compute_frobenius_norm(self, p, beta):
        """
        Computes the Frobenius norm of a 1D tensor based on the given beta value.

        Args:
            p (torch.Tensor): A 1D tensor.
            beta (float): A scaling factor for the Frobenius norm calculation.

        Returns:
            torch.Tensor: The Frobenius norm as a tensor.
        """
        # Ensure p is a 1D tensor
        p = p.flatten()

        # Compute sums A and B using torch
        A = torch.sum(p ** 2)
        B = torch.sum(p ** 3)

        # Compute the squared Frobenius norm
        frobenius_norm_squared = beta ** 2 * (A - 2 * B + A ** 2)

        # Compute the Frobenius norm
        frobenius_norm = torch.sqrt(frobenius_norm_squared)

        return frobenius_norm

    def compute_jacobian_frobenius_norm(self, softmax_matrices):
        """
        Computes the Frobenius norm of the Jacobian of the softmax matrix for each sample and head.

        Args:
            softmax_matrices (torch.Tensor): Tensor of shape [3, 12, 197, 197], where each [197, 197]
                                             is a softmax matrix for a given sample and head.

        Returns:
            torch.Tensor: A tensor of shape [3, 12], where each entry corresponds to the Frobenius norm
                          of the Jacobian for the respective sample and head.
        """
        # Ensure input is a PyTorch tensor
        softmax_matrices = torch.tensor(softmax_matrices, dtype=torch.float32)

        # Compute softmax squared
        softmax_squared = softmax_matrices ** 2

        # Compute diagonal terms: softmax[i, i]² * (1 - softmax[i, i])²
        diag_terms = softmax_squared * (1 - softmax_matrices) ** 2
        diag_sum = torch.sum(diag_terms, dim=(-2, -1))  # Sum over last two dimensions

        # Compute off-diagonal terms: softmax[i, j]² * softmax[j, i]²
        off_diag_sum = torch.sum(softmax_squared @ softmax_squared.transpose(-2, -1), dim=(-2, -1)) - diag_sum

        # Frobenius norm squared
        frobenius_norm_squared = diag_sum + off_diag_sum

        # Return Frobenius norm
        return torch.sqrt(frobenius_norm_squared)
