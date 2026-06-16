from dotenv import load_dotenv
from pathlib import Path

def load_env(filename: str = "dev.env"):
    # Load project .env if present
    env_path = Path(__file__).resolve().parents[1] / filename
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        raise FileNotFoundError(f"Env file not found: {env_path}")
