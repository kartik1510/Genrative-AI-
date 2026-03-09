# Encoder--Decoder CNN for Image-to-Image Translation

This project implements a **baseline Encoder--Decoder Convolutional
Neural Network (CNN)** to perform **image reconstruction** using the
CIFAR-10 dataset.

Instead of classifying images, the model learns to **recreate an image
after compressing it into a latent representation**.

------------------------------------------------------------------------

# Project Objective

The goal of this experiment is to:

-   Load paired images from the dataset
-   Normalize images to **\[-1, 1\]**
-   Train an **Encoder--Decoder CNN**
-   Use **MSE / L1 reconstruction loss**
-   Visualize reconstructed output images

This experiment demonstrates why simple reconstruction models often
generate **blurry images**.

------------------------------------------------------------------------

# Dataset

This project uses the **CIFAR-10 dataset**.

  Property          Value
  ----------------- ---------
  Total Images      60,000
  Training Images   50,000
  Test Images       10,000
  Image Size        32 × 32
  Channels          RGB
  Classes           10

Example classes include airplanes, cars, birds, cats, and trucks.

For this experiment, **class labels are not used** because the task
focuses on **image reconstruction rather than classification**.

------------------------------------------------------------------------

# Model Architecture

The model follows an **Encoder → Decoder** pipeline.

Input Image → Encoder (CNN Layers) → Latent Representation → Decoder →
Reconstructed Image

### Encoder

Extracts important features and compresses the image using convolutional
layers.

Typical layers: - Conv2D - ReLU activation

### Decoder

Reconstructs the image from compressed features.

Typical layers: - ConvTranspose2D - ReLU - Tanh (final output layer)

------------------------------------------------------------------------

# Loss Function

The network is trained using **reconstruction loss**, which measures the
difference between the original image and the reconstructed image.

### Mean Squared Error (MSE)

Measures squared differences between predicted and actual pixel values.

### L1 Loss

Measures absolute differences between predicted and actual pixel values.

------------------------------------------------------------------------

# Training Process

Training steps:

1.  Load images from CIFAR-10
2.  Normalize images to \[-1,1\]
3.  Pass images through the encoder
4.  Decode latent features back into images
5.  Compute reconstruction loss
6.  Update model weights using **Adam optimizer**

------------------------------------------------------------------------

# Results

After training, the model reconstructs images similar to the original
input.\
However, reconstructed images often appear **blurry**.

This occurs because pixel-based loss functions encourage the model to
predict **average pixel values**, which smooths fine details.

------------------------------------------------------------------------

# Technologies Used

-   Python
-   PyTorch
-   Torchvision
-   Matplotlib
-   Jupyter Notebook

------------------------------------------------------------------------

# Installation

Clone the repository:

git clone
https://github.com/yourusername/encoder-decoder-image-translation.git

Move to the project directory:

cd encoder-decoder-image-translation

Install dependencies:

pip install torch torchvision matplotlib

Run the notebook or script:

jupyter notebook

------------------------------------------------------------------------

# Key Learning

This experiment demonstrates:

-   How encoder--decoder CNN architectures work
-   The concept of latent representations
-   Why reconstruction loss produces blurry outputs
-   The motivation behind advanced models like **GANs and Diffusion
    models**

------------------------------------------------------------------------

# Author

Kartik\
Computer Science Engineering Student
