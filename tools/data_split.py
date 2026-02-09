from pathlib import  Path
from sklearn.preprocessing import LabelEncoder
import  numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def data_split():
    root = Path(__file__).parent.parent
    # Paths to read and write
    root_dir = root / "data" / "proccessed" / "Overlay_dataset"
    save_dir = root / "data" / "train_test_data"

    # Make sure save_dir exists
    save_dir.mkdir(parents=True, exist_ok=True)

    path = Path(root_dir)

    # Load all images
    image_paths = list(root_dir.glob("*/*.png"))
    print("Number of images in the dataset:", len(image_paths))

    # Extract labels from folder names
    labels = [img_path.parent.name for img_path in image_paths]

    print("-------------------- Classes Distribution in the dataset --------------------------")
    print("Number of images from Benign",labels.count("benign") )
    print("Number of images from normal",labels.count("normal") )
    print("Number of images from malignant",labels.count("malignant"))


    # Label encoding
    label_encode = LabelEncoder()
    target_labels = label_encode.fit_transform(labels)
    image_paths = np.array(image_paths)


    x_train, x_test, y_train, y_test = train_test_split(
        image_paths,
        target_labels,
        test_size=0.2,        # 20% test
        random_state=42,
        shuffle=True,
        stratify=target_labels  # VERY important for classification
    )
    # create dataframes
    train_df = pd.DataFrame({
        "image_path": x_train,
        "label": y_train
    })

    test_df = pd.DataFrame({
        "image_path": x_test,
        "label": y_test
    })

    # save CSV
    train_df.to_csv(Path(save_dir) / "train.csv", index=False)
    test_df.to_csv(Path(save_dir) / "test.csv", index=False)

    print("\nSaved:")
    print(Path(save_dir) / "train.csv")
    print(Path(save_dir) / "test.csv")

data_split()