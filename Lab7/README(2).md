# 🎨 Neural Style Transfer (NST)
### CSET419 — Introduction to Generative AI | Lab 7

---

## 📌 What is Neural Style Transfer?

Neural Style Transfer (NST) is a deep learning technique that takes **two images** —
a **content image** and a **style image** — and generates a **new image** that preserves
the structure of the content image while applying the artistic texture of the style image.

> Example: Take a photo of a city + Van Gogh's *Starry Night* → Output looks like the city painted by Van Gogh.

---

## 🧠 How It Works

```
Content Image  ──┐
                  ├──► VGG19 Feature Extractor ──► Loss Functions ──► Optimise Pixels ──► Generated Image
Style Image    ──┘
```

1. A pretrained **VGG19** CNN extracts features from both images
2. **Content Loss** ensures the generated image keeps the original structure
3. **Style Loss** (via Gram Matrix) ensures the texture/style is transferred
4. The generated image pixels are **optimised** to minimise total loss

---

## 📁 Project Structure

```
Lab7_NST/
│
├── Lab7_NST.ipynb               # Main Colab notebook (24 cells)
├── neural_style_transfer.py     # Standalone Python script
├── README.md                    # This file
│
├── images/
│   ├── content.jpg              # Input content image
│   └── style.jpg                # Input style image
│
└── nst_outputs/
    ├── step_0000.png            # Generated image at step 0
    ├── step_0050.png            # Generated image at step 50
    ├── step_0100.png            # Generated image at step 100
    ├── step_0150.png            # Generated image at step 150
    ├── step_0200.png            # Generated image at step 200
    ├── step_0250.png            # Generated image at step 250
    ├── final_output.png         # ✅ Final stylised image
    ├── comparison.png           # Content vs Style vs Generated
    ├── loss_curves.png          # Total / Content / Style loss graphs
    └── progress.png             # Evolution of image over steps
```

---

## 🚀 How to Run

### Option A — Google Colab (Recommended)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `Lab7_NST.ipynb`
3. Enable GPU: `Runtime → Change runtime type → T4 GPU`
4. Run all cells: `Runtime → Run All`
5. Download results:
```python
import shutil
from google.colab import files
shutil.make_archive('Lab7_Complete', 'zip', 'nst_outputs')
files.download('Lab7_Complete.zip')
```

---

### Option B — Local Machine

**Step 1 — Install dependencies:**
```bash
pip install torch torchvision pillow matplotlib
```

**Step 2 — Run with sample images (auto-downloaded):**
```bash
python neural_style_transfer.py
```

**Step 3 — Run with your own images:**
```bash
python neural_style_transfer.py \
  --content your_photo.jpg \
  --style   your_painting.jpg \
  --steps   300 \
  --alpha   1.0 \
  --beta    1e6
```

---

## ⚙️ Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--content` | sample | Path to content image |
| `--style` | sample | Path to style image |
| `--steps` | 300 | Number of optimisation steps |
| `--alpha` | 1.0 | Content loss weight |
| `--beta` | 1e6 | Style loss weight |
| `--save_every` | 50 | Save image every N steps |

---

## 🏗️ Model Architecture

| Component | Detail |
|-----------|--------|
| Base Model | VGG19 (pretrained on ImageNet) |
| Weights | Frozen — never updated |
| Content Layer | `conv4_2` |
| Style Layers | `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` |
| Optimiser | L-BFGS |
| What gets optimised | Pixel values of the generated image |

---

## 📐 Loss Functions

### Content Loss
Measures structural difference between generated and content images:
```
L_content = 0.5 × mean( (F_generated − F_content)² )
```

### Gram Matrix
Captures style/texture by computing feature correlations:
```
G[i,j] = Σ F[i,k] × F[j,k]  /  (C × H × W)
```

### Style Loss
Compares Gram matrices across 5 style layers:
```
L_style = Σ  w_l × MSE( Gram(generated)_l ,  Gram(style)_l )
```

### Total Loss
```
L_total = alpha × L_content  +  beta × L_style
```

---

## 📊 Expected Output

After running, you should observe:

- ✅ **Content preserved** — the subject/structure of the content image is recognisable
- ✅ **Style transferred** — brushstrokes, colours, texture of the style image are applied
- ✅ **Loss decreasing** — all three loss curves decrease over optimisation steps
- ✅ **Artistic result** — output looks like the content painted in the style image's art style

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `HTTPError 403` | Use `requests` with browser User-Agent headers |
| `FileNotFoundError: images/content.jpg` | Run the image generation cell first |
| `RuntimeError: Can't call numpy() on Tensor that requires grad` | Use `.detach()` before `.numpy()` |
| Very slow on CPU | Enable GPU in Colab: `Runtime → Change runtime type → T4 GPU` |
| Out of memory | Reduce `IMAGE_SIZE` from 512 to 256 |

---

## 📦 Dependencies

```
torch
torchvision
pillow
matplotlib
numpy
```

Install all at once:
```bash
pip install torch torchvision pillow matplotlib numpy
```

---

## 📚 References

1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *A Neural Algorithm of Artistic Style.* Journal of Vision.
2. Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR 2015.
3. [PyTorch VGG19 Documentation](https://pytorch.org/vision/stable/models/vgg.html)
4. [PyTorch NST Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

---

## 👨‍💻 Author

**Course:** CSET419 — Introduction to Generative AI
**Lab:** Week 7 — Neural Style Transfer
**Institution:** ___________________________
**Student Name:** ___________________________
**Date:** ___________________________

---

> *"A Neural Algorithm of Artistic Style"* — Gatys et al., 2015
