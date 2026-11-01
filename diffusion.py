import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data import get_dataloader

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
    alphas_cumprod = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine", device: torch.device = None):
        self.timesteps = timesteps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas.to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)

        # Forward process math: q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Reverse sampling math: q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, model: nn.Module, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = model(x_noisy, t)

        return F.mse_loss(predicted_noise, noise)

def save_forward_visualization(diffusion: GaussianDiffusion, image: torch.Tensor, save_path: str = "forward_diffusion.png"):
    timesteps_to_show = [0, 50, 100, 200, 400, 600, 800, 999]
    noisy_images = []

    noise = torch.randn_like(image.unsqueeze(0))
    for t_val in timesteps_to_show:
        t = torch.tensor([t_val], device=diffusion.device, dtype=torch.long)
        if t_val == 0:
            x_t = image.unsqueeze(0)
        else:
            x_t = diffusion.q_sample(image.unsqueeze(0), t, noise=noise)
        
        # Denormalize to [0, 1]
        x_t = (x_t * 0.5 + 0.5).clamp(0.0, 1.0)
        noisy_images.append(x_t.squeeze(0).cpu())

    grid = make_grid(noisy_images, nrow=len(timesteps_to_show), padding=2)
    grid_np = grid.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(16, 2.5))
    ax.imshow(grid_np)
    ax.set_xticks([(i + 0.5) * 66 for i in range(len(timesteps_to_show))])
    ax.set_xticklabels([f"t={t}" for t in timesteps_to_show], fontsize=11)
    ax.set_yticks([])
    ax.set_title("Forward Diffusion Process: q(x_t | x_0)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = GaussianDiffusion(timesteps=1000, schedule="cosine", device=device)

    loader = get_dataloader(dataset_key="stanford_cars", batch_size=1)
    sample_car = next(iter(loader))[0].to(device)

    save_forward_visualization(diffusion, sample_car, save_path="forward_diffusion.png")
    print("Saved forward_diffusion.png")
