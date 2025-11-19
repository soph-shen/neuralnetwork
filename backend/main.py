#builtin
from pathlib import Path
import random

#external
from PIL import Image
import numpy as np

#internal
from network import NeuralNetwork
from components.losses.loss_function import CrossEntropy

def load_image(path: Path) -> np.ndarray:
    """ Load one image and return it as a (1,4096) array in [0,1]"""
    img = Image.open(path).convert("L").resize((64, 64))
    array = np.array(img, dtype=np.float32)   
    normalize = array / 255.0                 #normalize after removing color in convert("L"), 
    flattened_array = normalize.flatten().reshape(1, -1)
    return flattened_array

def label_data():
    """
    Label each training data to its true value. 
    File name already corresponds to each value, i.e. if b=1, label=0
    """
    digits = [0,1,2,3,4,5,6,7,8,9]
    image_info = []
    for c in range(1, 101):
        for b in range(1, 11):
            for x_idx, label in enumerate(digits, start=1):
                path = f"data/input_{c}_{b}_{x_idx}.jpg"
                image_info.append((path, label))
    return image_info

def build_dataset(batch):
    """Converts to features X: (N,4096) and labels y:1-D array (N,)"""
    X_list, y_list = [], []
    for path, label in batch:
        path = Path(path)
        if not path.exists():
            continue
        X_list.append(load_image(path))
        y_list.append(label)
    
    if not X_list:
        return None, None 
    
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y

def iter_image_batches(image_info, batch_size):
    """Creates mini batches"""
    for i in range(0, len(image_info), batch_size):
        batch = image_info[i:i+batch_size]
        X, y = build_dataset(batch)
        if X is None:
            continue
        yield X, y

def class_distribution(image_info, name="dataset"):
    """Checks label distribution for imbalanced data"""
    labels = [label for _, label in image_info]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution for {name}:")
    print(dict(zip(unique, counts)))

def train_model(image_info, batch_size=32, num_epochs=5):
    first_path = Path(image_info[0][0])
    D = load_image(first_path).shape[1]

    loss_fn = CrossEntropy()
    net = NeuralNetwork(
        dimensions=[D, 256, 128, 10],   #2 hidden layers to learn better
        learning_rate=0.01,
        loss_function=loss_fn,
    )

    for epoch in range(num_epochs):
        random.shuffle(image_info)  

        batches = 0
        for X_batch, y_batch in iter_image_batches(image_info, batch_size):
            net.train(X_batch, y_batch)
            batches += 1

        if (epoch + 1) % 1 == 0:
            sample = image_info[: min(512, len(image_info))]
            X_s, y_s = next(iter_image_batches(sample, len(sample)))
            probs = net.predict(X_s)
            loss = loss_fn.get_test_loss(y_s, probs)
            print(f"Epoch {epoch+1}/{num_epochs}  batches:{batches}  loss:{loss:.4f}")

    net.save_network("trained_network.json")
    return net

def test_model(image_info):
    X, y = build_dataset(image_info)
    loss_fn = CrossEntropy()
    net = NeuralNetwork.load_network("trained_network.json", loss_function=loss_fn)

    probs = net.predict(X)
    predicted_digits = np.argmax(probs, axis=1)

    accuracy = np.mean(predicted_digits == y)
    print("True labels:     ", y)
    print("Predicted labels:", predicted_digits)
    print(f"Accuracy: {accuracy:.3f}")

    return predicted_digits, accuracy

def main():
    all_info = label_data()
    random.shuffle(all_info)
    split = int(0.8 * len(all_info))
    train_image_info = all_info[:split]
    val_image_info   = all_info[split:]

    class_distribution(train_image_info, "train")
    class_distribution(val_image_info, "val")

    train_model(train_image_info, batch_size=32, num_epochs=30)
    print("Validation performance:")
    test_model(val_image_info)

if __name__ == "__main__":
    main()