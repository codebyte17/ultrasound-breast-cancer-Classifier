import kagglehub
import os
import shutil

from src.config import load_yaml


def downloader():
    data = load_yaml('data_config.yaml')

    # Download latest version
    path = kagglehub.dataset_download(data.dataset.kaggle_path)

    print("Path to dataset files:", path)

    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(data.dataset.raw_data_dir), exist_ok=True)

    # Move dataset
    shutil.move(path, data.dataset.raw_data_dir)

    print("Dataset moved to:", data.dataset.raw_data_dir)
    return {"status": "SUCCESS",
            "location" : data.dataset.raw_data_dir}

downloader()