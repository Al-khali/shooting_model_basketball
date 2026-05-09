"""Application settings — loaded from environment variables or .env file."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    # App
    app_name: str = "AI Shoot"
    debug: bool = False
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    # Inference
    pose_model: str = "yolov11-pose"  # "yolov11-pose" | "vitpose" | "mediapipe"
    device: str = "cpu"  # "cpu" | "cuda" | "mps"
    inference_batch_size: int = 4

    # VLM
    vlm_provider: str = "gemini"  # "gemini" | "openai" | "qwen"
    vlm_model: str = "gemini-1.5-flash"
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # Data paths
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    data_reference_dir: str = "data/reference"
    models_dir: str = "models"


settings = Settings()
