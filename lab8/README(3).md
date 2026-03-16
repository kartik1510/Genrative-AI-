# CSET419 – Introduction to Generative AI
## Lab 8: Create Artistic Outputs using Neural Art Concepts

---

## 📌 Objective
Generate artistic images using Generative Adversarial Networks (GANs) by exploring the latent space of the generator. Investigate how different latent vectors and GAN architectures produce diverse and creative visual outputs.

---

## 🧠 GAN Architectures Used

| Type | Architecture | Purpose |
|---|---|---|
| Basic GAN | **DCGAN** (Deep Convolutional GAN) | Generate artistic images from random noise |
| Advanced GAN | **CycleGAN** | Unpaired image-to-image style translation |

---

## 📂 Dataset
- **CIFAR-10** — 60 000 colour images across 10 classes (32×32 px)
- A **5 000-image subset** is used by default for fast training
- Downloaded automatically via `torchvision.datasets.CIFAR10`

---

## ⚙️ Requirements

```bash
pip install torch torchvision matplotlib numpy pillow
```

| Library | Version |
|---|---|
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.12 |
| torchvision | ≥ 0.13 |
| matplotlib | ≥ 3.5 |
| numpy | ≥ 1.21 |

---

## 🚀 How to Run

### Option 1 – Jupyter Notebook (Local)
```bash
jupyter notebook CSET419_Lab_8.ipynb
```
Run all cells top to bottom (`Kernel → Restart & Run All`).

### Option 2 – Google Colab (Recommended for GPU)
1. Upload `CSET419_Lab_8.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Click **Run All**

---

## ⏱️ Estimated Training Time

| Mode | Image Size | Epochs | Dataset | Time |
|---|---|---|---|---|
| **Fast (default)** | 32×32 | 5 | 5 000 | ~2–3 min CPU / ~30s GPU |
| **Full quality** | 64×64 | 30 | 50 000 | ~4–5 hrs CPU / ~25 min GPU |

To switch to full quality, change these lines in the notebook:
```python
IMG_SIZE  = 64
EPOCHS    = 30
MAX_IMGS  = 50000
NGF = NDF = 64
```

---

## 📋 Lab Tasks

### Task 1 – Data Preparation
- Load CIFAR-10 dataset
- Normalise pixel values to **[−1, 1]**
- Define latent vector dimension (`Z_DIM = 100`)

### Task 2 – Load / Define GAN Models
- **DCGAN**: 3-layer transposed conv generator + 3-layer conv discriminator
- **CycleGAN**: ResNet generator (4 residual blocks) + PatchGAN discriminator

### Task 3 – Latent Space Exploration
- Generate **10 artistic samples** from random latent vectors
- Perform **latent vector interpolation** between z₁ and z₂ (10 steps)
- Observe smooth visual morphing across the latent space

### Task 4 – Generate Artistic Outputs
- **Evolution grid** — generator output quality across epochs
- **Diversity grid** — 25 unique images from different z vectors
- **Real vs Generated** comparison
- **CycleGAN** domain translation demo (A → B)

---

## 📁 Output Files

| File | Description |
|---|---|
| `dcgan_samples.png` | 10 random artistic samples |
| `dcgan_interpolation.png` | Latent space interpolation strip |
| `dcgan_evolution.png` | Output quality across epochs |
| `dcgan_diversity.png` | 25 unique generated images |
| `real_vs_generated.png` | Side-by-side comparison |
| `cyclegan_demo.png` | CycleGAN domain translation |

---

## 🏗️ Model Architecture

### DCGAN Generator (32×32 fast mode)
```
z (100×1×1)
  → ConvTranspose2d → BN → ReLU   # 4×4
  → ConvTranspose2d → BN → ReLU   # 8×8
  → ConvTranspose2d → BN → ReLU   # 16×16
  → ConvTranspose2d → Tanh         # 32×32 RGB
```

### DCGAN Discriminator
```
RGB (3×32×32)
  → Conv2d → LeakyReLU             # 16×16
  → Conv2d → BN → LeakyReLU       # 8×8
  → Conv2d → BN → LeakyReLU       # 4×4
  → Conv2d → Sigmoid               # scalar probability
```

### CycleGAN Generator
```
RGB → ReflectionPad → Conv → InstanceNorm → ReLU
    → Downsample ×2
    → ResidualBlock ×4
    → Upsample ×2
    → ReflectionPad → Conv → Tanh
```

---

## 📊 Expected Observations
- New images generated purely from random noise vectors
- Artistic variations from different latent vectors
- Smooth visual transitions during interpolation
- Creative patterns not copied from training images

---

## 🎓 Learning Outcomes
After completing this lab, students will understand:
- How GAN models generate creative visual outputs
- The importance of latent space in Generative AI
- Differences between DCGAN and CycleGAN architectures
- How generative models enable AI-based artistic content creation

---

## 👤 Course Info
**Course:** CSET419 – Introduction to Generative AI  
**Lab:** 8  
**Topic:** Neural Art with GANs
