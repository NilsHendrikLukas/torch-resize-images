""" Script to resize images using PyTorch.
"""
import os
import argparse
from math import ceil, floor
from multiprocessing.context import Process

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm
import GPUtil as GPUtil


def parse_args():
    """Parses cmd arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True,
                        help="Path to the input directory.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Path to the input directory.")
    parser.add_argument("--width", type=int,
                        default=224)
    parser.add_argument("--height", type=int,
                        default=224)
    parser.add_argument('-n', "--n_procs", type=int,
                        default=1)
    parser.add_argument('--no-center-crop', action='store_false', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument("--gpu", help="Which GPU to run on. If none is provided, least active will be chosen. ")

    return parser.parse_args()


def pick_gpu():
    """
    Picks a GPU with the least memory load.
    :return:
    """
    try:
        gpu = GPUtil.getFirstAvailable(order='memory', maxLoad=2, maxMemory=0.8, includeNan=False,
                                       excludeID=[], excludeUUID=[])[0]
        return gpu
    except Exception as e:
        print(e)
        return "0"


def reserve_gpu(mode_or_id):
    """ Chooses a GPU.
    If None, uses the GPU with the least memory load.
    """
    if mode_or_id:
        gpu_id = mode_or_id
        os.environ["CUDA_VISIBLE_DEVICES"] = mode_or_id
    else:
        gpu_id = str(pick_gpu())
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Selecting GPU id {gpu_id}")


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


def main():
    args = parse_args()
    print(f"Resize Images: {args}")

    reserve_gpu(args.gpu)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    resize_images_parallel(args.root, output_folder=args.output_dir, overwrite=args.overwrite, n_procs=args.n_procs,
                           resize_kwargs={
                               "width": args.width,
                               "height": args.height,
                               "center_crop": not args.no_center_crop
                           })


if __name__ == "__main__":
    main()
