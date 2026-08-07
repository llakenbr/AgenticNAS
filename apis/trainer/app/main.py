from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import torch
import os

# Paths inside container (mounted to local machine)
DATA_DIR = "/app/data"
LOGS_DIR = "/app/logs"

app = FastAPI(title="trainer",
              description="API for training NN models",
              version="0.0.1")

print("Started trainer API running torch version:", torch.__version__)

class TrainRequest(BaseModel):
    epochs: int
    learning_rate: float
    model_name: str

@app.post("/train")
def start_training(request: TrainRequest):
    # Run training as a subprocess so the API stays responsive
    subprocess.Popen([
        "python", "train.py",
        f"--epochs={request.epochs}",
        f"--lr={request.learning_rate}",
        f"--model={request.model_name}"
    ])
    return {"status": "test"}