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
    app_name: str = Field(default="agent-memory", validation_alias="APP_NAME")
    app_description: str = Field(
        default="FastAPI service for structured agent episodic and summary memory",
        validation_alias="APP_DESCRIPTION",
    )
    app_version: str = Field(default="0.1.0")
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_dir: str = Field(default="/app/logs", validation_alias="LOG_DIR")

    postgres_host: str = Field(default="n8n_postgres", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="agent_memory", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="agent_memory_user", validation_alias="POSTGRES_USER")
    postgres_password: Optional[str] = Field(default=None, validation_alias="POSTGRES_PASSWORD")
    postgres_password_file: Optional[str] = Field(
        default=None,
        validation_alias="POSTGRES_PASSWORD_FILE",
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
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password_resolved}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
