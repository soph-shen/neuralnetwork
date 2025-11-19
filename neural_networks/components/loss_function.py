# external
import numpy as np

class LossFunction:
    """Superclass / interface for loss function"""
    # returns the base gradient
    def get_training_loss(self, response: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # returns a scalar evaluation of the loss (mean cross-entropy)
    def get_test_loss(self, response: np.ndarray, prediction: np.ndarray) -> float:
        raise NotImplementedError
    

class CrossEntropy(LossFunction):
    def get_training_loss(self, response: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        pred = np.atleast_2d(prediction)
        N = pred.shape[0]

        one_hot = np.zeros_like(pred)
        one_hot[np.arange(N), response] = 1

        grad = (pred - one_hot) / N
        return grad
    
    def get_test_loss(self, response: np.ndarray, prediction: np.ndarray) -> float:
        pred = np.atleast_2d(prediction)
        N = pred.shape[0]
        py = pred[np.arange(N), response]                #selects only probability value assigned to correct class
        log_likelihood = -np.log(np.clip(py, 1e-12, 1 - 1e-12))     #ensures probability is never 0 or 1
        return float(np.mean(log_likelihood))