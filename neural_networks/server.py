# builtin
from typing import List

# external 
from PIL import Image, ImageFilter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# internal
from network import NeuralNetwork
from components.loss_function import CrossEntropy


#Setup app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Predict64(BaseModel):
    pixels: List[List[float]]  # 2D list 64x64
        
# Load model
MODEL_PATH = "trained_network.json"
_loss_fn = CrossEntropy()
_net = NeuralNetwork.load_network(MODEL_PATH, loss_function=_loss_fn)

def preprocess_canvas(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float32)

    if arr.shape != (64, 64):
        raise ValueError(f"Expected 64x64 pixels, got {arr.shape}")

    if arr.max() == 0:          #return 0 if nothing drawn
        return arr

    arr = np.clip(arr, 0.0, 1.0)        #in case of noise, clip to [0,1] so that it can be used in the model
    
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))   #larger radius = stronger blur

    # stretch contrast so strokes really stand out: scale min to 0 and max to 1
    norm = np.array(img, dtype=np.float32) / 255.0
    mn, mx = norm.min(), norm.max()
    norm = norm - mn
    if mx - mn > 1e-6:
        norm = norm / (mx - mn)

    return norm

# Routes
@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Predict64):
    #Receive 64x64 canvas from frontend, preprocess it, run the model, and return predictions
    try:
        arr = np.array(payload.pixels, dtype=np.float32)
        arr_proc = preprocess_canvas(arr)

        #Sanity check
        print(
            " Input: min", arr_proc.min(),
            "max", arr_proc.max(),
            "sum", arr_proc.sum()
        )

        X = arr_proc.flatten().reshape(1, -1)
        probs = _net.predict(X)
        print("Probs:", probs[0])

        pred = int(np.argmax(probs, axis=1)[0])
        return {"predicted_digit": pred, "probs": probs[0].tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
