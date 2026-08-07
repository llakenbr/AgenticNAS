# ./app/models/request.py
from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.settings import settings


DATASET_NAME_OPTIONS = [
    str(dataset["name"])
    for dataset in settings.dataset_options["dataset_options"]
    if "name" in dataset
]


class zeroEstimateParameters(BaseModel):
    batch_size: int | None = None
    output_nodes: int
    num_imgs_or_batches: int | None = None
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


class zeroEstimateRequest(BaseModel):
    parameters: zeroEstimateParameters
    model_spec: dict[str, Any] | str  # Accept either a dict or a JSON string for flexibility
