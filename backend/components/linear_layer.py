'''Note: z = Wa + b with activation so it turns into a(next) = f(Wa + b)
'''
#external
import numpy as np

#internal
from components.activation_function import ActivationFunction


class LinearLayer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationFunction,
        weights: np.ndarray = None,
        biases: np.ndarray = None,
    ):
        """Fully-connected layer: z = XW + b, a = activation(z)."""

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Parameter initialization
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.randn(input_size, output_size).astype(np.float32) * np.sqrt(2.0 / input_size)

        if biases is not None:
            self.biases = biases
        else:
            self.biases = np.zeros((1, output_size), dtype=np.float32)

        self.last_x: np.ndarray | None = None 
        self.last_z: np.ndarray | None = None   

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        input: (batch_size, input_size) or (input_size,)
        return: (batch_size, output_size)
        """
        x = np.atleast_2d(input)   
        self.last_x = x
        self.last_z = x @ self.weights + self.biases
        return self.activation.apply(self.last_z)

    def backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass.

        gradient: dC/da for this layer, shape (N, output_size)
        returns: dC/d(input) for previous layer, shape (N, input_size)
        """
        if self.last_x is None or self.last_z is None:
            raise RuntimeError("forward must be called before backward")

        grad = np.atleast_2d(gradient)
        x = np.atleast_2d(self.last_x)
        z = np.atleast_2d(self.last_z)

        dz = self.activation.apply_gradient(z, grad)
        if dz is None:
            raise RuntimeError("activation.apply_gradient returned None – check activation class")
        dz = np.atleast_2d(dz)

        dW = x.T @ dz                   
        db = np.sum(dz, axis=0, keepdims=True) 

        grad_prev = dz @ self.weights.T   

        # 4) Update parameters
        self.__update_weights(dW, learning_rate)
        self.__update_biases(db, learning_rate)

        return grad_prev

    def __update_weights(self, gradient: np.ndarray, learning_rate: float):
        self.weights -= learning_rate * gradient

    def __update_biases(self, gradient: np.ndarray, learning_rate: float):
        self.biases -= learning_rate * gradient

    # return in order of (weight, bias)
    def save_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        return self.weights, self.biases
