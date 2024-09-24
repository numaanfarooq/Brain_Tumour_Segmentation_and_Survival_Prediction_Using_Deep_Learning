import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Directly use the indexed paths """
        image_path = self.images_path[index]
        mask_path = self.masks_path[index]

        # Check if paths are strings
        if not isinstance(image_path, str):
            raise TypeError(f"Expected string for image path, got {type(image_path)}")
        if not isinstance(mask_path, str):
            raise TypeError(f"Expected string for mask path, got {type(mask_path)}")

        # Debugging: print paths
        print(f"Loading image from: {image_path}")
        print(f"Loading mask from: {mask_path}")

        # Check if paths exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path {image_path} does not exist")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask path {mask_path} does not exist")

        """ Reading image """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = image / 255.0  # Normalize
        image = np.transpose(image, (2, 0, 1))  # Convert to channel-first
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  # Normalize
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
