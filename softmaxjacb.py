import numpy as np

class SoftmaxJacobianFrobeniusNorm:
    """
    Class to compute the Frobenius norm of the Jacobian of the softmax function.
    Optionally stores the softmax probabilities.
    """

    def __init__(self, beta=1.0, store_probabilities=False):
        """
        Initialize the class with a scaling factor beta.

        Parameters:
        - beta (float): Scaling factor applied before the softmax function.
        - store_probabilities (bool): Whether to store softmax probabilities.
        """
        self.beta = beta
        self.store_probabilities = store_probabilities
        self.probabilities = None  # Initialize probabilities as None

    def compute(self, x):
        """
        Compute the Frobenius norm of the Jacobian of the softmax function
        given the input vector x, assuming x is already softmaxed.

        Parameters:
        - x (numpy.ndarray): Input vector of shape (n,), already softmaxed.

        Returns:
        - frobenius_norm (float): Frobenius norm of the Jacobian.
        """
        # Ensure x is a 1D numpy array and treat it as already-softmaxed probabilities
        p = np.asarray(x).flatten()

        # Optionally store the probabilities
        if self.store_probabilities:
            self.probabilities = p

        # Compute sums A and B based on the already-softmaxed probabilities
        A = np.sum(p ** 2)
        B = np.sum(p ** 3)

        # Compute the squared Frobenius norm using the formula:
        # ||J||_F^2 = beta^2 * (A - 2B + A^2)
        frobenius_norm_squared = self.beta ** 2 * (A - 2 * B + A ** 2)

        # Compute the Frobenius norm
        frobenius_norm = np.sqrt(frobenius_norm_squared)

        return frobenius_norm
