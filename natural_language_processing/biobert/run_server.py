import uvicorn
from pydantic import BaseSettings


class UvicornSettings(BaseSettings):
    root_path: str = '/biobert/api'
    port: int = 8000

    class Config:
        env_prefix = 'api_'


if __name__ == "__main__":
    settings = UvicornSettings()
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, root_path=settings.root_path)
