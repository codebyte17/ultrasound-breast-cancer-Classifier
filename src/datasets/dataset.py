import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# build custom dataset and dataloader
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.images = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        images = Image.open(self.images[index]).convert("RGB")

        # Convert to Tensor
        images = transforms.ToTensor()(images)
        # Resize the image size
        resize_transform = transforms.Resize((224, 224))
        # Apply the transform
        resized_img = resize_transform(images)
        # labels
        labels = torch.tensor(self.labels[index])
        return resized_img, labels