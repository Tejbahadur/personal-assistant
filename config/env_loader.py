import os
from dotenv import load_dotenv

load_dotenv()

def get_env(key: str, default: str = None) -> str:
    """Safely retrieve environment variables."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} not found and no default provided.")
    return value
