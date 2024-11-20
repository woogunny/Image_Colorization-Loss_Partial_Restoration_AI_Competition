from PIL import Image
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    def __init__(self, damage_dir, origin_dir, transform=None):
        self.damage_dir = damage_dir
        self.origin_dir = origin_dir
        self.transform = transform
        self.damage_files = sorted(os.listdir(damage_dir))
        self.origin_files = sorted(os.listdir(origin_dir))

        assert len(self.damage_files) == len(self.origin_files), "The number of images in gray and color folders must match"

    def __len__(self):
        return len(self.damage_files)

    def __getitem__(self, idx):
        damage_img_name = self.damage_files[idx]
        origin_img_name = self.origin_files[idx]

        damage_img_path = os.path.join(self.damage_dir, damage_img_name)
        origin_img_path = os.path.join(self.origin_dir, origin_img_name)

        damage_img = Image.open(damage_img_path).convert("RGB")
        origin_img = Image.open(origin_img_path).convert("RGB")

        if self.transform:
            damage_img = self.transform(damage_img)
            origin_img = self.transform(origin_img)

        return {'A': damage_img, 'B': origin_img}

