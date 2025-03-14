import torch
from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader
import config
import numpy as np

class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform
        
        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images))
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)
    
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        zebra_image = self.zebra_images[index % self.zebra_len]
        horse_image = self.horse_images[index % self.horse_len]
        
        zebra_path = os.path.join(self.root_zebra, zebra_image)
        horse_path = os.path.join(self.root_horse, horse_image)
        
        zebra_image = np.array(Image.open(zebra_path).convert("RGB"))
        horse_image = np.array(Image.open(horse_path).convert("RGB"))
        
        if self.transform:
            augmentations = self.transform(image=zebra_image, image0=horse_image)
            zebra_image = augmentations["image"]
            horse_image = augmentations["image0"]
            
        return zebra_image, horse_image
        
        
        
        
        

