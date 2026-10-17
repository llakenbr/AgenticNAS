# ./app/models/request.py
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .request import trainingSpec


class trainModel(BaseModel):
    training_spec: trainingSpec
    train_loss: list[float] = Field(default_factory=list)
    torch_model: Any
    train_dataloader: Any
    test_dataloader: Any
    torch_device: Any

