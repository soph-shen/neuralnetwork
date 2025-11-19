# Chinese Digit Classifier

This project lets you draw a Chinese digit on a canvas and classifies it using a neural network implemented from scratch using forward pass, backpropagation, and cross entropy. The network was trained on MNIST data from https://www.kaggle.com/datasets/gpreda/chinese-mnist/code.

## Structure

- `backend/` – FastAPI server and neural network code (Python, Poetry)
- `frontend/` – React + Vite frontend (TypeScript)
- `data/` – Training images

## Running the project

### Frontend
```cd frontend
npm install
npm run dev
```

### Backend

```bash
poetry install
poetry run uvicorn backend.server:app --reload
