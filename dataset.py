import numpy as np
import cv2
import random

from skimage import io, transform, filters

import torch
import torch.utils.data
from torchvision import datasets, models, transforms


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = io.imread(img_path)
        mask = io.imread(mask_path)

        image = image.astype('float32') / 255
        mask = mask.astype('float32') / 255

        if self.aug:
            if random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                image = image[::-1, :, :].copy()
                mask = mask[::-1, :].copy()
            # if random.uniform(0, 1) > 0.5:
            #     image=transform.rotate(image, 10).copy()
            #     mask=transform.rotate(mask, 10).copy()
            # if random.uniform(0, 1) > 0.5:
            #     image=transform.rotate(image, 20).copy()
            #     mask=transform.rotate(mask, 20).copy()
            # if random.uniform(0, 1) > 0.5:
            #     image=transform.rotate(image, 30).copy()
            #     mask=transform.rotate(mask, 30).copy()
            # if random.uniform(0, 1) > 0.5:
            #     image=transform.rotate(image, 40).copy()
            #     mask=transform.rotate(mask, 40).copy()
            #if random.uniform(0, 1) > 0.5:
            #     image=filters.gaussian(image,sigma=1).copy()
            #     mask=mask.copy()
        image = image.transpose((2, 0, 1))
        mask = mask[:,:,np.newaxis]
        mask = mask.transpose((2, 0, 1))

        return image, mask
