ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
VOICE_ID=your_default_voice_id_here
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ASSEMBLYAI_API_KEY: str | None = None
    ELEVENLABS_API_KEY: str | None = None
    VOICE_ID: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()
