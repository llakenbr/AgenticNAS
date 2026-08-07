# ./core/settings.py
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = Field(default="model-trainer")
    app_description: str = Field(
        default="API for training neural architecture models",
        validation_alias="APP_DESCRIPTION",
    )
    app_version: str = Field(default="1.0.0")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_dir: str = Field(default="/app/logs", validation_alias="LOG_DIR")

    dataset_options: ClassVar[dict[str, list[dict[str, object]]]] = {
        "dataset_options": [
            {
                "name": "cifar10",
                "dataloader": "cifar",
                "input_shape": ["batch_size", 3, 32, 32],
            },
            {
                "name": "cifar100",
                "dataloader": "cifar",
                "input_shape": ["batch_size", 3, 32, 32],
            },
            {
                "name": "svhn",
                "dataloader": "cifar",
                "input_shape": ["batch_size", 3, 32, 32],
            },
            {
                "name": "ImageNet16-120",
                "dataloader": "cifar",
                "input_shape": ["batch_size", 3, 16, 16],
            },
            {
                "name": "ImageNet1k",
                "dataloader": "cifar",
                "input_shape": ["batch_size", 3, 224, 224],
            },
            {
                "name": "mnist",
                "dataloader": "mnist",
                "input_shape": ["batch_size", 1, 28, 28],
            },
        ]
    }


settings = Settings()