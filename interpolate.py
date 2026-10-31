import os
import math
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from model import get_model
from diffusion import GaussianDiffusion
from sample import ddim_sample

def slerp(z0: torch.Tensor, z1: torch.Tensor, alpha: float) -> torch.Tensor:
    z0_flat = z0.flatten()
    z1_flat = z1.flatten()

    z0_norm = z0_flat / torch.norm(z0_flat)
    z1_norm = z1_flat / torch.norm(z1_flat)

    dot = torch.sum(z0_norm * z1_norm).clamp(-0.9995, 0.9995)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    theta_t = theta_0 * alpha
    sin_theta_t = torch.sin(theta_t)

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    return (s0 * z0 + s1 * z1)

@torch.no_grad()
def ddim_sample_from_latent(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    latent: torch.Tensor,
    ddim_steps: int = 50,
    device: torch.device = None
) -> torch.Tensor:
    model.eval()
    device = device or next(model.parameters()).device
    x = latent.clone().to(device)
    n_samples = x.shape[0]

    times = torch.linspace(0, diffusion.timesteps - 1, ddim_steps + 1, dtype=torch.long).to(device)
    time_pairs = list(zip(times[:-1], times[1:]))

    for prev_t_val, t_val in reversed(time_pairs):
        t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)

        eps = model(x, t)

        alpha_cumprod_t = diffusion.alphas_cumprod[t_val]
        alpha_cumprod_prev = diffusion.alphas_cumprod[prev_t_val] if prev_t_val >= 0 else torch.tensor(1.0, device=device)

        pred_x0 = (x - torch.sqrt(1.0 - alpha_cumprod_t) * eps) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        dir_xt = torch.sqrt(1.0 - alpha_cumprod_prev) * eps
        x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt

    return (x * 0.5 + 0.5).clamp(0.0, 1.0)

def generate_morph_gif(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    frames: int = 30,
    ddim_steps: int = 50,
    save_path: str = "car_morph.gif",
    device: torch.device = None
):
    device = device or next(model.parameters()).device

    z0 = torch.randn(1, 3, 64, 64, device=device)
    z1 = torch.randn(1, 3, 64, 64, device=device)

    alphas = torch.linspace(0.0, 1.0, frames)
    # Loop back smoothly
    alphas = torch.cat([alphas, torch.linspace(1.0, 0.0, frames)])

    pil_frames = []
    print(f"Generating {len(alphas)} morph frames using DDIM ({ddim_steps} steps)...")
    for alpha in tqdm(alphas, desc="Interpolating"):
        z_interp = slerp(z0, z1, alpha.item())
        img_tensor = ddim_sample_from_latent(model, diffusion, z_interp, ddim_steps=ddim_steps, device=device)
        pil_img = to_pil_image(img_tensor.squeeze(0).cpu())
        pil_frames.append(pil_img)

    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=70,
        loop=0
    )
    print(f"Saved morph animation to: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/latest.pt")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=str, default="car_morph.gif")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device=device)
    diffusion = GaussianDiffusion(timesteps=1000, schedule="cosine", device=device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded trained weights from {args.checkpoint}")
    else:
        print(f"No checkpoint found at {args.checkpoint}, running demo on initialized weights.")

    generate_morph_gif(
        model=model,
        diffusion=diffusion,
        frames=args.frames,
        ddim_steps=args.steps,
        save_path=args.output,
        device=device
    )

if __name__ == "__main__":
    main()
