import os
import csv
import copy
import time
import argparse
import warnings
import torch
import torch.nn as nn
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pynvml as nvml
    nvml.nvmlInit()
    _nvml_handle = nvml.nvmlDeviceGetHandleByIndex(0)
    _nvml_ok = True
except Exception:
    _nvml_ok = False

def gpu_stats() -> dict:
    if not _nvml_ok:
        return {}
    mem   = nvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
    util  = nvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
    temp  = nvml.nvmlDeviceGetTemperature(_nvml_handle, nvml.NVML_TEMPERATURE_GPU)
    power = nvml.nvmlDeviceGetPowerUsage(_nvml_handle) / 1000.0  # mW -> W
    return {
        "gpu_util_%"   : util.gpu,
        "vram_used_gb" : round(mem.used  / 1024**3, 2),
        "vram_total_gb": round(mem.total / 1024**3, 2),
        "temp_c"       : temp,
        "power_w"      : round(power, 1),
    }

from torchvision.utils import save_image

from data import get_dataloader
from model import get_model
from diffusion import GaussianDiffusion
from sample import ddim_sample


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


def train(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp   = device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    log_path = os.path.join(args.checkpoint_dir, "loss_log.csv")
    if not args.resume or not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "loss", "time_s",
                "gpu_util_%", "vram_used_gb", "vram_total_gb",
                "temp_c", "power_w"
            ])

    loader    = get_dataloader(
        dataset_key=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    model     = get_model(image_size=args.image_size, device=device)
    ema       = EMA(model, decay=args.ema_decay)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, schedule=args.schedule, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    if args.compile and hasattr(torch, "compile"):
        try:
            import triton
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
        except Exception:
            print("Note: torch.compile skipped (Triton compiler is not present on Windows). Running high-performance CUDA.")

    start_epoch = 0
    if args.resume and os.path.exists(os.path.join(args.checkpoint_dir, "latest.pt")):
        ckpt = torch.load(os.path.join(args.checkpoint_dir, "latest.pt"), map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if use_amp else 0.0
    print(f"Device : {device} ({torch.cuda.get_device_name(0) if use_amp else 'CPU'})")
    print(f"VRAM   : {vram_gb:.1f} GB  |  AMP dtype: {amp_dtype}  |  Batch: {args.batch_size}")
    print(f"Dataset: {args.dataset}  |  Image size: {args.image_size}x{args.image_size}")
    print(f"Epochs : {args.epochs}  |  Timesteps: {args.timesteps}  |  Schedule: {args.schedule}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d}/{args.epochs:03d}", dynamic_ncols=True)
        for x0 in pbar:
            x0 = x0.to(device, non_blocking=True)
            t  = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=device).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = diffusion.compute_loss(model, x0, t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - start_time
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        stats = gpu_stats()
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                f"{avg_loss:.4f}",
                f"{elapsed:.1f}",
                stats.get("gpu_util_%",    ""),
                stats.get("vram_used_gb",  ""),
                stats.get("vram_total_gb", ""),
                stats.get("temp_c",        ""),
                stats.get("power_w",       ""),
            ])

        checkpoint = {
            "epoch"     : epoch,
            "model"     : model.state_dict(),
            "ema"       : ema.shadow.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "loss"      : avg_loss,
            "image_size": args.image_size,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "latest.pt"))

        if (epoch + 1) % args.save_every == 0:
            path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:03d}.pt")
            torch.save(checkpoint, path)
            print(f"Saved checkpoint → {path}")

        if (epoch + 1) % args.sample_every == 0 or epoch == 0:
            os.makedirs(args.sample_dir, exist_ok=True)
            samples = ddim_sample(ema.shadow, diffusion, n_samples=16, ddim_steps=50, device=device)
            preview_path = os.path.join(args.sample_dir, f"epochs001-{epoch+1:03d}.png")
            save_image(samples, preview_path, nrow=4, padding=2)
            print(f"Saved preview → {preview_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str,   default="stanford_cars")
    parser.add_argument("--image_size",     type=int,   default=64,   choices=[64, 128])
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--timesteps",      type=int,   default=1000)
    parser.add_argument("--schedule",       type=str,   default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--ema_decay",      type=float, default=0.9999)
    parser.add_argument("--save_every",     type=int,   default=25)
    parser.add_argument("--sample_every",   type=int,   default=10)
    parser.add_argument("--sample_dir",     type=str,   default="./progress")
    parser.add_argument("--checkpoint_dir", type=str,   default="./checkpoints")
    parser.add_argument("--num_workers",    type=int,   default=2)
    parser.add_argument("--compile",        action="store_true")
    parser.add_argument("--resume",         action="store_true")
    args = parser.parse_args()

    train(args)
