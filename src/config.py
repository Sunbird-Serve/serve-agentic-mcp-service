from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from typing import Optional, List, Tuple
class Settings(BaseSettings):
    KAFKA_BROKERS: str = "localhost:19092"
    TOPIC_WA_OUT: str = "serve.vm.whatsapp.out"
    TOPIC_WA_IN: str = "serve.vm.whatsapp.in"
    WA_ASCII_ONLY: bool = False
    SERVE_BASE_URL: str = "https://serve-v1.evean.net"
    SERVE_TIMEOUT_SECONDS: float = 15.0
    SERVE_DEFAULT_AGENCY_ID: str = "1-74f81200-dc16-4c65-bf7a-a3ab75952432"
    
    # WhatsApp Cloud API configuration
    WHATSAPP_ACCESS_TOKEN: str = ""
    WHATSAPP_PHONE_NUMBER_ID: str = ""
    WHATSAPP_API_VERSION: str = "v21.0"
    
    # Class video configuration
    SERVE_CLASS_VIDEO_PATH: str = "./media/serve_class_intro.mp4"
    SERVE_WELCOME_VIDEO_PATH: str = "./media/welcome.mp4"
    SERVE_THANKYOU_VIDEO_PATH: str = "./media/thankyou.mp4"
    WA_MEDIA_CACHE_PATH: str = "./media_cache.json"
    
    # Firebase configuration
    FIREBASE_SERVICE_ACCOUNT_JSON: str = "./serve-sandbox-firebase-adminsdk-4i44o-ac8df1245e.json"
    FIREBASE_PROJECT_ID: Optional[str] = None
    FIREBASE_RESET_CONTINUE_URL: Optional[str] = None
    FIREBASE_WEB_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
