# builtin
from __future__ import annotations
import json

# external
import numpy as np

# internal
from components.losses.loss_function import LossFunction
from components.linear_layer import LinearLayer
from components.activations.activation_function import ReLU, Softmax

class NeuralNetwork():
    def __init__(self, dimensions: list[int], learning_rate: float, loss_function: LossFunction):
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.layers: list[LinearLayer] = []
        
        for i in range(len(dimensions) - 1):  #Has N-1 total layers
            input_size = dimensions[i]
            output_size = dimensions[i + 1]

            # All hidden layers use ReLU, last layer uses Softmax
            if i < len(dimensions) - 2:
                activation = ReLU()
            else:
                activation = Softmax()

            layer = LinearLayer(input_size, output_size, activation)  #New layer with weights and biases
            self.layers.append(layer)
    
    @classmethod
    def load_network(cls, path: str, loss_function: LossFunction):
        with open(path, "r") as f:
            data = json.load(f)

        # Initialize a new network with the same structure
        net = cls(
            dimensions=data["dimensions"],
            learning_rate=data["learning_rate"],
            loss_function=loss_function
        )

        for layer, saved_layer in zip(net.layers, data["layers"]):
            cls._load_layer_parameters(layer, saved_layer)
            cls._load_layer_activation(layer, saved_layer["activation"])

        print(f"Network loaded from {path}")
        return net
        
    def predict(self, input: np.ndarray) -> np.ndarray:
        """Returns class probabilities for input data"""
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, input: np.ndarray, response: np.ndarray) -> np.ndarray:
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        prediction = x
        
        grad = self.loss_function.get_training_loss(response, prediction)
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.learning_rate)     #backpropagation
        return prediction
    
    def save_network(self, path: str):
        """Creates a JSON file to save the model with weights and biases"""
        data = {
            "dimensions": self.dimensions,
            "learning_rate": self.learning_rate,
            "layers": []
        }

        for layer in self.layers:
            layer_info = {
                "weights": layer.weights.tolist(),
                "biases": layer.biases.tolist(),
                "activation": layer.activation.__class__.__name__
            }
            data["layers"].append(layer_info)

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Network saved to {path}")
        
    @staticmethod
    def _load_layer_parameters(layer, saved_layer):
        layer.weights = np.array(saved_layer["weights"])
        layer.biases = np.array(saved_layer["biases"])
    
    def _load_layer_activation(layer, saved_layer):
        act_name = saved_layer
        if act_name == "ReLU":
            layer.activation = ReLU()
        elif act_name == "Softmax":
            layer.activation = Softmax()
        else:
            raise ValueError(f"Unknown activation type: {act_name}")