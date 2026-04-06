from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    DATABASE_URL: str = "mysql+pymysql://root:root@localhost:3306/fake_currency_detection"
    MODEL_PATH: str = "models/xception_currency_final.h5"
    MAX_BASE64_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_ORIGINS: str = "http://localhost:5173"
    ALLOWED_MIME_TYPES: str = "image/jpeg,image/png,image/webp"

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_db_url(cls, v):
        if not v.startswith(("mysql+pymysql://", "mysql://")):
            raise ValueError("DATABASE_URL must use mysql+pymysql:// scheme")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
