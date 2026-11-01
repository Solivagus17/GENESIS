import os
import argparse
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

from model import get_model
from diffusion import GaussianDiffusion
from data import get_dataloader

@torch.no_grad()
def ddpm_sample(model: nn.Module, diffusion: GaussianDiffusion, n_samples: int = 16, device: torch.device = None) -> torch.Tensor:
    model.eval()
    device = device or next(model.parameters()).device
    x = torch.randn(n_samples, 3, 64, 64, device=device)

    for i in tqdm(reversed(range(diffusion.timesteps)), desc="DDPM Sampling", total=diffusion.timesteps):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        eps = model(x, t)

        beta_t = diffusion.betas[i]
        alpha_t = diffusion.alphas[i]
        alpha_cumprod_t = diffusion.alphas_cumprod[i]

        if i > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(diffusion.posterior_variance[i])
        else:
            noise = 0.0
            sigma_t = 0.0

        mean = (1.0 / torch.sqrt(alpha_t)) * (x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)) * eps)
        x = mean + sigma_t * noise

    return (x * 0.5 + 0.5).clamp(0.0, 1.0)

@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    n_samples: int = 16,
    ddim_steps: int = 50,
    eta: float = 0.0,
    device: torch.device = None
) -> torch.Tensor:
    model.eval()
    device = device or next(model.parameters()).device
    x = torch.randn(n_samples, 3, 64, 64, device=device)

    times = torch.linspace(0, diffusion.timesteps - 1, ddim_steps + 1, dtype=torch.long).to(device)
    time_pairs = list(zip(times[:-1], times[1:]))

    for prev_t_val, t_val in tqdm(reversed(time_pairs), desc=f"DDIM Sampling ({ddim_steps} steps)", total=len(time_pairs)):
        t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        prev_t = torch.full((n_samples,), prev_t_val, device=device, dtype=torch.long)

        eps = model(x, t)

        alpha_cumprod_t = diffusion.alphas_cumprod[t_val]
        alpha_cumprod_prev = diffusion.alphas_cumprod[prev_t_val] if prev_t_val >= 0 else torch.tensor(1.0, device=device)

        # Estimate x_0
        pred_x0 = (x - torch.sqrt(1.0 - alpha_cumprod_t) * eps) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        # Compute variance parameter
        sigma = eta * torch.sqrt((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t) * (1.0 - alpha_cumprod_t / alpha_cumprod_prev))

        # Direction to x_t
        dir_xt = torch.sqrt(1.0 - alpha_cumprod_prev - sigma ** 2) * eps
        noise = torch.randn_like(x) if sigma > 0 else 0.0

        x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + sigma * noise

    return (x * 0.5 + 0.5).clamp(0.0, 1.0)

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT
        inception = models.inception_v3(weights=weights).to(device).eval()
        inception.fc = nn.Identity()
        self.inception = inception
        self.device = device
        self.resize = T.Resize((299, 299), antialias=True)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(self.resize(x))
        return self.inception(x)

def compute_statistics(features: np.ndarray):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

@torch.no_grad()
def evaluate_fid(model: nn.Module, diffusion: GaussianDiffusion, dataset_key: str = "stanford_cars", num_samples: int = 500, batch_size: int = 50) -> float:
    device = next(model.parameters()).device
    extractor = InceptionFeatureExtractor(device)
    loader = get_dataloader(dataset_key=dataset_key, batch_size=batch_size)

    # 1. Extract real image features
    real_features = []
    collected = 0
    print(f"Extracting features from {num_samples} real images...")
    for real_batch in loader:
        real_batch = (real_batch * 0.5 + 0.5).clamp(0.0, 1.0).to(device)
        feats = extractor(real_batch).cpu().numpy()
        real_features.append(feats)
        collected += real_batch.shape[0]
        if collected >= num_samples:
            break
    real_features = np.concatenate(real_features, axis=0)[:num_samples]

    # 2. Extract generated image features via DDIM
    gen_features = []
    collected = 0
    print(f"Generating and extracting features from {num_samples} fake images...")
    pbar = tqdm(total=num_samples, desc="FID Generation")
    while collected < num_samples:
        current_b = min(batch_size, num_samples - collected)
        fake_batch = ddim_sample(model, diffusion, n_samples=current_b, ddim_steps=50, device=device)
        feats = extractor(fake_batch).cpu().numpy()
        gen_features.append(feats)
        collected += current_b
        pbar.update(current_b)
    pbar.close()
    gen_features = np.concatenate(gen_features, axis=0)[:num_samples]

    mu_real, sigma_real = compute_statistics(real_features)
    mu_gen, sigma_gen = compute_statistics(gen_features)

    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return float(fid_score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/latest.pt")
    parser.add_argument("--method", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./progress")
    parser.add_argument("--dataset", type=str, default="stanford_cars")
    parser.add_argument("--eval_fid", action="store_true")
    parser.add_argument("--fid_samples", type=int, default=500)
    parser.add_argument("--use_ema", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device=device)
    diffusion = GaussianDiffusion(timesteps=1000, schedule="cosine", device=device)

    epoch = 0
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        key = "ema" if (args.use_ema and "ema" in ckpt) else "model"
        model.load_state_dict(ckpt[key])
        epoch = ckpt.get("epoch", 0) + 1
        print(f"Loaded {key} weights from {args.checkpoint} (epoch {epoch})")
    else:
        print(f"No checkpoint found at {args.checkpoint}, using initialized weights.")

    if args.method == "ddim":
        samples = ddim_sample(model, diffusion, n_samples=args.samples, ddim_steps=args.steps, device=device)
    else:
        samples = ddpm_sample(model, diffusion, n_samples=args.samples, device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = args.output if args.output else os.path.join(args.output_dir, f"epochs001-{epoch:03d}.png")

    nrow = int(np.sqrt(args.samples))
    save_image(samples, out_path, nrow=nrow, padding=2)
    print(f"Saved {args.samples} cars to {out_path}")

    if args.eval_fid:
        score = evaluate_fid(model, diffusion, dataset_key=args.dataset, num_samples=args.fid_samples)
        print(f"Fréchet Inception Distance (FID): {score:.2f}")

if __name__ == "__main__":
    main()
