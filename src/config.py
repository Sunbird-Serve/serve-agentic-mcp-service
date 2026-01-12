from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    KAFKA_BROKERS: str = "localhost:19092"
    TOPIC_WA_OUT: str = "serve.vm.whatsapp.out"
    TOPIC_WA_IN: str = "serve.vm.whatsapp.in"
    WA_ASCII_ONLY: bool = False
    SERVE_BASE_URL: str = "https://serve-v1.evean.net"
    
    # WhatsApp Cloud API configuration
    WHATSAPP_ACCESS_TOKEN: str = "EAAKvnZCuvi3QBQeg2qz9LqQrSPwmsTUJCSfZBlMgU9EoiHp9c1cR24UT2Od6y7gOPBuI6x4WGaC0V9qjZAfZBvBqLz50tGlCelttvsfqyZARgi1W2NTmEHMSwEWZAiKCjBxObNZASwTGZBuIMB8NBHG3dOVxVWlkodpPhH3mdzrZBQ9Nkzy6zZA7VDNXrwhGHbW76d3KkbIU6JJcYH6WuBkxCNfR4dpb2GfSJDqO0uVozBw2BRZBfyQknfWdV4ocRXqXHJtDw03ZCvZCvwkCvliT0sjXU"
    WHATSAPP_PHONE_NUMBER_ID: str = "882287331642834"
    WHATSAPP_API_VERSION: str = "v21.0"
    
    # Class video configuration
    SERVE_CLASS_VIDEO_PATH: str = "./media/serve_class_intro.mp4"
    WA_MEDIA_CACHE_PATH: str = "./media_cache.json"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
