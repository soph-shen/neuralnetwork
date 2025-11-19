#external
import numpy as np
class ActivationFunction:
    """Interface for all activation function """
    def apply(self, data: np.ndarray) -> np.ndarray:
        #Forward: a = f(z).
        raise NotImplementedError("apply() must be implemented in subclasses")

    def apply_gradient(self, data: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        #Backward: given z and dC/da, return dC/dz.
        raise NotImplementedError("apply_gradient() must be implemented in subclasses")

class ReLU(ActivationFunction):
    def apply(self, data: np.ndarray) -> np.ndarray:
        return np.maximum(0, data)

    def apply_gradient(self, data: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        data: z (pre-activation), same shape as gradient
        ReLU'(z) = 1 if z > 0, else 0. This ensures all signal passed to next layer is nonnegative
        """
        relu_derivative = (data > 0).astype(gradient.dtype)     #gradient (dC/da) that is passed back from the next layer, computes da/dz=ReLu'(z)
        return gradient * relu_derivative                       #spits out dC/dz

class Softmax(ActivationFunction):
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        data: logits z, shape (N, C)
        returns softmax probabilities, same shape.
        """
        e_x = np.exp(data - np.max(data, axis=1, keepdims=True))       #remove max to avoid overflow (number too large for computer to store)
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def apply_gradient(self, data: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        With softmax + cross-entropy, the loss already gives us dC/dz, so we just pass the gradient through unchanged.
        """
        return gradient