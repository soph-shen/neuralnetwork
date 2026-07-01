# Chinese Digit Classifier

Draw a Chinese digit on a canvas and watch a neural network guess what you wrote. The catch: I built the network from scratch, no ML libraries doing the heavy lifting.

<p align="center">
  <img src="demo_five.png" width="45%" alt="Drawing of the Chinese digit five, predicted correctly as 5">
  &nbsp;&nbsp;
  <img src="demo_eight.png" width="45%" alt="Drawing of the Chinese digit eight, predicted correctly as 8">
</p>

## What it does

You draw a digit in the browser, hit Predict, and the model tells you which digit it is. Above are two live examples: 五 (five) and 八 (eight), both classified correctly.

The interesting part is under the hood. Instead of reaching for PyTorch or TensorFlow, I implemented the whole network by hand: the forward pass, backpropagation, and cross entropy loss. It's trained on the [Chinese MNIST dataset](https://www.kaggle.com/datasets/gpreda/chinese-mnist/code).

## How it's built

The project has two halves that talk to each other:

* `backend/` is a FastAPI server (Python) that holds the neural network code and serves predictions.
* `frontend/` is a React and Vite app (TypeScript) with the drawing canvas.

When you click Predict, the canvas sends your drawing to the backend, the network runs a forward pass, and the predicted digit comes back to the screen.

## Running it yourself

You'll need the training data first. Download the [Chinese MNIST dataset](https://www.kaggle.com/datasets/gpreda/chinese-mnist/code) from Kaggle and place it in a `data/` folder. (The data is too large to store here, which is normal for datasets.)

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

**Backend**

```bash
poetry install
poetry run uvicorn backend.server:app --reload
```

## Why I built it

I wanted to really understand what happens inside a neural network, not just call `.fit()` and trust it. Writing the forward and backward passes by hand made the math click, and wrapping it in a draw to predict app turned it into something you can actually play with.
