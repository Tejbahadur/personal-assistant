import yaml
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()
