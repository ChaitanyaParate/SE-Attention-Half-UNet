import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# Retrive images and its corresponding masks
class TrainDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = os.listdir(image_dir)

        self.img = os.listdir(self.image_dir)
        self.mask = os.listdir(self.mask_dir)
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.img[index])
        mask_path = os.path.join(self.mask_dir, self.mask[index])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask /= 255.0
        mask = (mask > 0.5).astype(np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask