"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings
from pydantic import field_validator, ConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Database configuration
    DATABASE_URL: str = "mysql+pymysql://root:root@localhost:3306/fake_currency_detection"
    
    # Model configuration
    MODEL_PATH: str = "models/cnn_pytorch_best.pth"
    
    # API configuration
    MAX_BASE64_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    ALLOWED_MIME_TYPES: str = "image/jpeg,image/png,image/webp"
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    @field_validator("DATABASE_URL")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        """Validate database URL scheme."""
        valid_schemes = ("mysql+pymysql://", "mysql://", "sqlite:///")
        if not any(v.startswith(scheme) for scheme in valid_schemes):
            raise ValueError(
                f"DATABASE_URL must use one of: {', '.join(scheme + '...' for scheme in valid_schemes)}"
            )
        return v
    
    @field_validator("ALLOWED_ORIGINS")
    @classmethod
    def validate_origins(cls, v: str) -> str:
        """Validate origins format."""
        origins = v.split(",")
        for origin in origins:
            origin = origin.strip()
            if origin and not origin.startswith(("http://", "https://", "*")):
                raise ValueError(f"Invalid origin: {origin}. Must start with http://, https://, or *")
        return v


# Global settings instance
settings = Settings()
