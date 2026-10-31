import os
import sys
import glob
import tarfile
import urllib.request
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid

DATASET_CONFIGS = {
    "stanford_cars": {
        "name": "Stanford Cars (Cars 196)",
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/stanford-cars.tgz",
        "archive_name": "stanford-cars.tgz",
        "extract_folder": "stanford-cars",
        "image_subdirs": ["cars_train", "cars_test"],
        "license": "Non-commercial Academic Research",
        "official_url": "https://ai.stanford.edu/~jkrause/cars/car_dataset.html",
        "total_images": 16185,
        "resolution": "~700x500"
    },
    "cifar10": {
        "name": "CIFAR-10 (Automobile Class)",
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz",
        "archive_name": "cifar10.tgz",
        "extract_folder": "cifar10",
        "image_subdirs": ["train/automobile", "test/automobile"],
        "license": "MIT License",
        "official_url": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "total_images": 6000,
        "resolution": "32x32 RGB"
    }
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_dataset(dataset_key: str, data_dir: str = "./data") -> Path:
    config = DATASET_CONFIGS[dataset_key]
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    archive_path = data_path / config["archive_name"]
    extract_path = data_path / config["extract_folder"]
    
    if not extract_path.exists():
        if not archive_path.exists():
            print(f"Downloading {config['name']} from {config['url']}")
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(config["url"], headers=headers)
            
            with urllib.request.urlopen(req) as response:
                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=config['archive_name']) as t:
                    with open(archive_path, 'wb') as out_file:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            t.update(len(chunk))

        print(f"Extracting {archive_path.name} to {data_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            if dataset_key == "cifar10":
                members = [m for m in tar.getmembers() if "automobile" in m.name]
                tar.extractall(path=data_path, members=members)
            else:
                tar.extractall(path=data_path)
    return extract_path

class CarDataset(Dataset):
    def __init__(self, dataset_key: str = "stanford_cars", data_dir: str = "./data", image_size: int = 64):
        super().__init__()
        self.dataset_key = dataset_key
        self.config = DATASET_CONFIGS[dataset_key]
        self.image_size = image_size
        self.extract_path = download_dataset(dataset_key, data_dir)
        
        cache_file = Path(data_dir) / f"{dataset_key}_{image_size}.pt"
        if cache_file.exists():
            print(f"Loading cached {dataset_key} from {cache_file}...")
            self.data = torch.load(cache_file)
        else:
            image_paths = []
            for subdir in self.config["image_subdirs"]:
                target_dir = self.extract_path / subdir
                if target_dir.exists():
                    for ext in ("*.jpg", "*.jpeg", "*.png"):
                        image_paths.extend(list(target_dir.glob(ext)))
                        
            if len(image_paths) == 0:
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    image_paths.extend(list(self.extract_path.rglob(ext)))

            image_paths.sort()
            if len(image_paths) == 0:
                raise RuntimeError(f"No car images found in {self.extract_path}")

            print(f"Caching {len(image_paths)} images at {image_size}x{image_size} to RAM...")
            crop_resize = T.Compose([T.Resize(image_size), T.CenterCrop(image_size)])
            tensors = []
            for p in tqdm(image_paths, desc="Processing"):
                try:
                    img = Image.open(p).convert("RGB")
                    img = crop_resize(img)
                    tensors.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))
                except Exception:
                    continue

            self.data = torch.stack(tensors)
            torch.save(self.data, cache_file)
            print(f"Saved cache ({self.data.shape[0]} images) to {cache_file}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx]
        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])
        # Normalize [0, 255] uint8 -> [-1.0, 1.0] float32
        return (img.float() / 127.5) - 1.0

def get_dataloader(dataset_key: str = "stanford_cars", data_dir: str = "./data", batch_size: int = 64, image_size: int = 64, num_workers: int = 0):
    dataset = CarDataset(dataset_key=dataset_key, data_dir=data_dir, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    return loader

def save_sample_batch(batch_tensor: torch.Tensor, save_path: str = "sample_batch.png"):
    imgs = (batch_tensor * 0.5 + 0.5).clamp(0.0, 1.0)
    grid = make_grid(imgs, nrow=8, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.title(f"Processed Batch: {batch_tensor.shape[0]} images ({batch_tensor.shape[2]}x{batch_tensor.shape[3]}) in [-1, 1]", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="stanford_cars", choices=["stanford_cars", "cifar10"])
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    loader = get_dataloader(dataset_key=args.dataset, batch_size=args.batch_size)
    batch = next(iter(loader))
    save_sample_batch(batch, save_path=f"sample_batch_{args.dataset}.png")
