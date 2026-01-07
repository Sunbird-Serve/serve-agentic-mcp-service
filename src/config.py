from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    KAFKA_BROKERS: str = "localhost:19092"
    TOPIC_WA_OUT: str = "serve.vm.whatsapp.out"
    TOPIC_WA_IN: str = "serve.vm.whatsapp.in"
    WA_ASCII_ONLY: bool = False
    SERVE_BASE_URL: str = "https://serve-v1.evean.net"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
