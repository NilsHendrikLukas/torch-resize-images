import os
from math import ceil, floor
from multiprocessing.context import Process

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm


class ResizeInputDataset(Dataset):
    def __init__(self, root, transform=None, split: float = 1.0):
        self.root = root
        self.transform = transform

        all_items = os.listdir(self.root)
        n = len(all_items)

        lower, upper = split

        self.items = all_items[floor(n*lower):min(ceil(n*upper), n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.items[idx]
        image = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image, img_name


def compose_transformations(width: int, height: int, center_crop: bool):
    """ Composes a transformation that keeps the size ratio.
    """
    transformations = [
        transforms.Resize(max(width, height), interpolation=Image.BICUBIC),
        transforms.CenterCrop((width, height)),
        transforms.ToTensor()
    ]

    if not center_crop:
        del transformations[1]

    return transforms.Compose(transformations)


def resize_images(input_folder: str, output_folder: str, overwrite: bool, resize_kwargs: dict,
                  split=(0.0, 1.0), index=0):
    ds = ResizeInputDataset(root=input_folder, transform=compose_transformations(**resize_kwargs), split=split)

    for img, img_name in tqdm(ds, desc="Resizing Images", disable=index != 0):
        if os.path.exists(os.path.join(output_folder, img_name)):
            if overwrite:
                save_image(img, os.path.join(output_folder, img_name))


def resize_images_parallel(input_folder: str, output_folder: str, overwrite: bool, n_procs: int, resize_kwargs: dict):
    split = 1.0 / n_procs

    processes = []
    for m in range(n_procs):
        this_split = (split * m, split * (m+1))

        p = Process(target=resize_images,
                    args=(input_folder, output_folder, overwrite, resize_kwargs, this_split, m))
        p.daemon = True
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print("Done Resizing!")