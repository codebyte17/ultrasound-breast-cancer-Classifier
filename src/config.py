from pathlib import Path
import yaml


class Config:
    """Convert dict to dot-access object"""
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)

            setattr(self, key, value)


def load_yaml(file_name: str):
    # __file__ = src/config.py
    SRC_DIR = Path(__file__).resolve().parent

    # go to project root
    ROOT_DIR = SRC_DIR.parent

    # configs directory
    CONFIG_DIR = ROOT_DIR / "configs"

    config_path = CONFIG_DIR / file_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)





