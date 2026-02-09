from dataset import CustomDataset
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def data_loader():
    root = Path(__file__).parent.parent

    # Paths to read and write
    train_dir = root / "data" / "train_test_data" / "train.csv"
    test_dir = root / "data" / "train_test_data" / "test.csv"

    train_dataset = pd.read_csv("data/train_test_data/train.csv")
    test_dataset = pd.read_csv("data/train_test_data/test.csv")

    # Keep file paths as strings
    x_train = train_dataset.iloc[:, 0].tolist()  # file paths as list of strings
    y_train = train_dataset.iloc[:, -1].values   # labels as numpy array

    x_test = test_dataset.iloc[:, 0].tolist()    # file paths as list of strings
    y_test = test_dataset.iloc[:, -1].values     # labels as numpy array

    # Optional: check shapes and samples

    print("First X_train sample:", x_train[0])
    print("First y_train sample:", y_train[0])


    x_train_dataset = CustomDataset(x_train,y_train)
    x_test_dataset = CustomDataset(x_test,y_test)

    x_train_dataloader = DataLoader(x_train_dataset,batch_size=32,shuffle=True,drop_last=True)
    x_test_dataloader = DataLoader(x_test_dataset,batch_size=32,shuffle=False,drop_last=True)


    print(" Number of batch from training data : ",x_train_dataloader.__len__())
    print(" Number of batch from testing data : ",x_test_dataloader.__len__())

    return x_train_dataloader, x_test_dataloader
