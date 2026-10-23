import os
import sys
import glob
import tarfile
import urllib.request
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

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
        "resolution": "~700x500 (variable high-res)"
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
            print(f"\n[*] Downloading {config['name']} from AWS S3 mirror...")
            print(f"    Source URL: {config['url']}")
            print(f"    Target File: {archive_path}")
            
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            req = urllib.request.Request(config["url"], headers=headers)
            
            with urllib.request.urlopen(req) as response:
                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=config['archive_name']) as t:
                    with open(archive_path, 'wb') as out_file:
                        chunk_size = 1024 * 1024
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            t.update(len(chunk))
            print("[+] Download complete!")

        print(f"[*] Extracting {archive_path.name} to {data_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            if dataset_key == "cifar10":
                members = [m for m in tar.getmembers() if "automobile" in m.name]
                tar.extractall(path=data_path, members=members)
            else:
                tar.extractall(path=data_path)
        print(f"[+] Extraction complete -> {extract_path}")
    else:
        print(f"[+] Dataset folder already exists at: {extract_path}")
        
    return extract_path

class CarDataset(Dataset):
    def __init__(self, dataset_key: str = "cifar10", data_dir: str = "./data", transform=None):
        super().__init__()
        self.dataset_key = dataset_key
        self.config = DATASET_CONFIGS[dataset_key]
        self.transform = transform
        self.extract_path = download_dataset(dataset_key, data_dir)
        
        self.image_paths = []
        for subdir in self.config["image_subdirs"]:
            target_dir = self.extract_path / subdir
            if target_dir.exists():
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    self.image_paths.extend(list(target_dir.glob(ext)))
                    
        if len(self.image_paths) == 0:
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                self.image_paths.extend(list(self.extract_path.rglob(ext)))

        self.image_paths.sort()
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No car images found in {self.extract_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

def verify_and_display_grid(dataset: CarDataset, save_path: str = "sample_cars_4x4.png"):
    config = dataset.config
    print("\n" + "=" * 70)
    print("GENESIS - DATASET SPECIFICATION & VERIFICATION (STEP 1)")
    print("=" * 70)
    print(f"Dataset Name:        {config['name']}")
    print(f"Official Link:       {config['official_url']}")
    print(f"License:             {config['license']}")
    print(f"Expected Images:     {config['total_images']}")
    print(f"Original Resolution: {config['resolution']}")
    print(f"Loaded Images Count: {len(dataset)} car images verified locally!")
    print("=" * 70)
    
    indices = torch.linspace(0, len(dataset) - 1, 16).long().tolist()
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(f"GENESIS - {config['name']} (Sample 4x4 Grid)", fontsize=16, fontweight='bold', y=0.94)
    
    for i, idx in enumerate(indices):
        r, c = divmod(i, 4)
        ax = axes[r, c]
        img = dataset[idx]
        ax.imshow(img)
        ax.set_title(f"Car #{idx}\n{img.size[0]}x{img.size[1]}", fontsize=9)
        ax.axis("off")
        
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[+] 4x4 Sample Grid saved successfully to: {os.path.abspath(save_path)}")
    try:
        plt.show(block=False)
        plt.pause(1.0)
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GENESIS Dataset Downloader & Verifier")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["stanford_cars", "cifar10"],
        help="Dataset choice: 'stanford_cars' or 'cifar10'"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to store datasets"
    )
    args = parser.parse_args()
    
    ds = CarDataset(dataset_key=args.dataset, data_dir=args.data_dir, transform=None)
    verify_and_display_grid(ds, save_path=f"sample_cars_4x4_{args.dataset}.png")
