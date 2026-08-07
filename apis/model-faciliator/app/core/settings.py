# ./core/settings.py
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = Field(default="model-faciliator")
    app_description: str = Field(
        default="API for interacting with nn_core model building and validation",
        validation_alias="APP_DESCRIPTION",
    )
    app_version: str = Field(default="1.1.0")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_dir: str = Field(default="/app/logs", validation_alias="LOG_DIR")

settings = Settings()
