# ./core/settings.py
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _read_secret(value: Optional[str], file_path: Optional[str]) -> Optional[str]:
    if file_path:
        path = Path(file_path)
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return value


class Settings(BaseSettings):
    app_name: str = Field(default="agent-runtime", validation_alias="APP_NAME")
    app_description: str = Field(
        default="FastAPI runtime for LangGraph-based agents",
        validation_alias="APP_DESCRIPTION",
    )
    app_version: str = Field(default="0.1.1", validation_alias="APP_VERSION")
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_dir: str = Field(default="/app/logs", validation_alias="LOG_DIR")

    postgres_host: str = Field(default="postgres", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="agent_runtime", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="agent_runtime_user", validation_alias="POSTGRES_USER")
    postgres_password: Optional[str] = Field(default=None, validation_alias="POSTGRES_PASSWORD")
    postgres_password_file: Optional[str] = Field(
        default=None,
        validation_alias="POSTGRES_PASSWORD_FILE",
    )

    azure_openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias="AZURE_OPENAI_API_KEY",
    )
    azure_openai_api_key_file: Optional[str] = Field(
        default=None,
        validation_alias="AZURE_OPENAI_API_KEY_FILE",
    )

    azure_openai_endpoint: Optional[str] = Field(
        default=None,
        validation_alias="AZURE_OPENAI_API_ENDPOINT",
    )
    azure_openai_endpoint_file: Optional[str] = Field(
        default=None,
        validation_alias="AZURE_OPENAI_API_ENDPOINT_FILE",
    )

    azure_openai_api_version: str = Field(
        default="2024-12-01-preview",
        validation_alias="AZURE_OPENAI_API_VERSION",
    )

    azure_openai_deployment: Optional[str] = Field(
        default=None,
        validation_alias="AZURE_OPENAI_DEPLOYMENT",
    )
    azure_openai_deployment_file: Optional[str] = Field(
        default=None,
        validation_alias="AZURE_OPENAI_DEPLOYMENT_FILE",
    )

    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def postgres_password_resolved(self) -> str:
        return _read_secret(self.postgres_password, self.postgres_password_file) or ""

    @property
    def azure_openai_api_key_resolved(self) -> Optional[str]:
        return _read_secret(self.azure_openai_api_key, self.azure_openai_api_key_file)

    @property
    def azure_openai_endpoint_resolved(self) -> Optional[str]:
        return _read_secret(self.azure_openai_endpoint, self.azure_openai_endpoint_file)

    @property
    def azure_openai_deployment_resolved(self) -> Optional[str]:
        return _read_secret(self.azure_openai_deployment, self.azure_openai_deployment_file)

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password_resolved}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_graph_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password_resolved}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()