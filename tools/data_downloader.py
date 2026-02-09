import kagglehub
import os
import shutil

def downloader():
    # Download latest version
    path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")

    print("Path to dataset files:", path)


    # Your desired destination folder
    dst_path = "./data/raw/breast_ultrasound"   # change this

    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Move dataset
    shutil.move(path, dst_path)

    print("Dataset moved to:", dst_path)
    return {"status": "SUCCESS",
            "location" : dst_path}
