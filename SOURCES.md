# Sources & References

A clear, straightforward log of where all the datasets, research papers, download mirrors, and tools used in **GENESIS** came from.

---

## 1. Datasets

### Stanford Cars (Cars 196) — *Primary Dataset*
* **What it is:** 16,185 high-resolution car images covering 196 distinct makes and models (e.g., BMW M3, Tesla Model S, 1993 Ford pickup). Originally introduced by Jonathan Krause and Fei-Fei Li at Stanford in 2013 (*"3D Object Representations for Fine-Grained Categorization"*).
* **Official page:** [ai.stanford.edu/~jkrause/cars/car_dataset.html](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* **The download issue & our mirror:** Stanford's original server (`ai.stanford.edu`) frequently refuses connections or times out, which commonly breaks PyTorch's built-in `torchvision.datasets.StanfordCars`. To solve this, we fetch the identical archive hosted on fast.ai's public AWS S3 bucket:  
  [https://s3.amazonaws.com/fast-ai-imageclas/stanford-cars.tgz](https://s3.amazonaws.com/fast-ai-imageclas/stanford-cars.tgz)
* **License:** Free for research, education, and non-commercial personal projects.
* **Image stats:** 16,185 images total (8,144 train / 8,041 test), variable dimensions averaging around 700×500 px.

---

### CIFAR-10 ("automobile" class) — *Fast Fallback*
* **What it is:** The car subset from Alex Krizhevsky's classic CIFAR-10 benchmark (2009).
* **Official page:** [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
* **Why we use it:** At 160 MB and 32×32 resolution, it lets us test our entire pipeline, loss curves, and sampler locally in minutes before committing to longer runs on the full 2 GB Stanford dataset.
* **Our mirror:** Also pulled from fast.ai's S3 mirror ([https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz](https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz)) where `data.py` extracts just the 6,000 car images.
* **License:** MIT License.

---

## 2. Research Papers & Architecture Design

Every piece of math and model code in this project is based on these foundational papers:

1. **Denoising Diffusion Probabilistic Models (DDPM)**  
   *Jonathan Ho, Ajay Jain, Pieter Abbeel (NeurIPS 2020)*  
   [Paper link (arXiv:2006.11239)](https://arxiv.org/abs/2006.11239)  
   *What we took from it:* The core diffusion formulation — defining the forward noising process $q(x_t | x_0)$, training the U-Net to predict the injected noise $\epsilon$ using simple MSE loss, and the ancestral reverse sampling loop.

2. **Improved Denoising Diffusion Probabilistic Models (Cosine Noise Schedule)**  
   *Alex Nichol, Prafulla Dhariwal (ICML 2021)*  
   [Paper link (arXiv:2102.09672)](https://arxiv.org/abs/2102.09672)  
   *What we took from it:* The cosine variance schedule. Ho et al.'s original linear schedule destroys low-frequency details too quickly on lower resolutions (32×32 and 64×64), making images look like flat static too early. The cosine schedule keeps noise gentle in the beginning and end, preserving much better car outlines.

3. **Denoising Diffusion Implicit Models (DDIM - Fast 50-step Sampler)**  
   *Jiaming Song, Chenlin Meng, Stefano Ermon (ICLR 2021)*  
   [Paper link (arXiv:2010.02502)](https://arxiv.org/abs/2010.02502)  
   *What we took from it:* Deterministic sampling. Instead of taking 1,000 random steps to generate a single car, DDIM skips steps along a deterministic trajectory so we can generate clean cars in 50 steps. It also makes latent-space interpolation possible (morphing Car A into Car B).

4. **U-Net Architecture & Attention for Diffusion**  
   *Prafulla Dhariwal, Alex Nichol — "Diffusion Models Beat GANs on Image Synthesis" (NeurIPS 2021)*  
   [Paper link (arXiv:2105.05233)](https://arxiv.org/abs/2105.05233)  
   *What we took from it:* The modern diffusion U-Net design:
   * Sinusoidal timestep embeddings projected into each residual block.
   * `GroupNorm` (32 groups) instead of `BatchNorm` (crucial because small batch sizes on an 8 GB card cause BatchNorm statistics to collapse).
   * Multi-head self-attention only at lower spatial resolutions (16×16 and 8×8) so the model captures global car geometry without blowing up memory.

5. **Fréchet Inception Distance (FID)**  
   *Martin Heusel et al. (NeurIPS 2017)*  
   [Paper link (arXiv:1706.08500)](https://arxiv.org/abs/1706.08500)  
   *What we took from it:* The metric we use in Step 6 to objectively score our generated cars against real ones.

---

## 3. Libraries & Environment

* **PyTorch & Torchvision:** Core deep learning framework, tensors, autograd, and GPU kernels. (Tested with CUDA 12.8 on an NVIDIA RTX 5050 8 GB).
* **Pillow (PIL):** High-speed image loading and basic cropping.
* **Matplotlib:** Rendering training progression grids and inspection plots.
* **tqdm:** Progress bars for data downloads, epoch training, and diffusion steps.
* **Gradio:** Lightweight web interface for generating cars interactively (added in Step 7).
