# ./app/models/request.py
from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.settings import settings


DATASET_NAME_OPTIONS = [
    str(dataset["name"])
    for dataset in settings.dataset_options["dataset_options"]
    if "name" in dataset
]


class trainingSpec(BaseModel):
    batch_size: int | None = None
    loss_function: str
    optimizer: str
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0005
    nesterov: bool = False
    ending_lr: float | None = None
    lr_schedule: str | None = None
    epochs: int
    dataset_name: str = Field(
        ...,
        json_schema_extra={"enum": DATASET_NAME_OPTIONS},
    )

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, value: str) -> str:
        if value not in DATASET_NAME_OPTIONS:
            available = ", ".join(DATASET_NAME_OPTIONS)
            raise ValueError(
                f"Invalid dataset_name '{value}'. Allowed values: {available}"
            )
        return value

class trainRequest(BaseModel):
    training_parameters: trainingSpec
    model_spec: dict[str, Any] | str  # Accept either a dict or a JSON string for flexibility
