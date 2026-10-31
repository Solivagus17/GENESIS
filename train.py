import os
import copy
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from tqdm import tqdm

from data import get_dataloader
from model import get_model
from diffusion import GaussianDiffusion

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

@torch.no_grad()
def sample_images(model: nn.Module, diffusion: GaussianDiffusion, n_samples: int = 16, device: torch.device = None) -> torch.Tensor:
    model.eval()
    device = device or next(model.parameters()).device
    x = torch.randn(n_samples, 3, 64, 64, device=device)

    for i in reversed(range(diffusion.timesteps)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        predicted_noise = model(x, t)

        beta_t = diffusion.betas[i]
        alpha_t = diffusion.alphas[i]
        alpha_cumprod_t = diffusion.alphas_cumprod[i]

        if i > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(diffusion.posterior_variance[i])
        else:
            noise = 0.0
            sigma_t = 0.0

        # Mean calculation
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise)
        x = mean + sigma_t * noise

    model.train()
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    loader = get_dataloader(dataset_key=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    model = get_model(device=device)
    ema = EMA(model, decay=args.ema_decay)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, schedule=args.schedule, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 0
    if args.resume and os.path.exists(os.path.join(args.checkpoint_dir, "latest.pt")):
        ckpt = torch.load(os.path.join(args.checkpoint_dir, "latest.pt"), map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    print(f"Training on: {device} | Mixed Precision bf16: {use_bf16} | Batch size: {args.batch_size}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d}/{args.epochs:03d}")
        for x0 in pbar:
            x0 = x0.to(device)
            t = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=device).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = diffusion.compute_loss(model, x0, t)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ema.update(model)

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:03d} complete | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.shadow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "latest.pt"))

        # Save sample grid
        if (epoch + 1) % args.sample_interval == 0 or epoch == 0:
            sample_grid = sample_images(model, diffusion, n_samples=16, device=device)
            grid_path = os.path.join(args.sample_dir, f"epoch_{epoch+1:03d}.png")
            save_image(sample_grid, grid_path, nrow=4, padding=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["stanford_cars", "cifar10"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", type=str, default="cosine")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--sample_interval", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--sample_dir", type=str, default="./samples")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(args)
