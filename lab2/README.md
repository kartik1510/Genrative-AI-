# Lab 2: Basic GAN for Image Generation

## Course

CSET419 – Introduction to Generative AI

## Objective

The objective of this lab is to design, train, and evaluate a **basic Generative Adversarial Network (GAN)** capable of generating synthetic images similar to a given dataset. This lab simulates a real-world scenario where missing images must be recreated to test an AI pipeline end-to-end.

---

## Problem Statement

A university’s digital archive server has partially crashed, resulting in the loss of several scanned images. Before restoring the archive, synthetic images are required to validate downstream AI systems. To address this, a GAN model is trained to generate realistic-looking images based on an existing dataset.

---

## Dataset

One dataset is used for training (chosen by the user):

* **MNIST** – Handwritten digits (28×28 grayscale)
* **Fashion-MNIST** – Clothing items (28×28 grayscale)

The dataset is loaded using PyTorch/Torchvision utilities.

---

## Input Parameters

The following configuration parameters are defined before training:

* `dataset_choice`: `mnist` or `fashion`
* `epochs`: Number of training epochs (recommended: 30–100)
* `batch_size`: Batch size (recommended: 64 or 128)
* `noise_dim`: Dimension of random noise vector (e.g., 50 or 100)
* `learning_rate`: Learning rate (e.g., 0.0002)
* `save_interval`: Interval (in epochs) for saving generated images

---

## Model Architecture

### Generator

* Takes a random noise vector as input
* Uses fully connected layers to transform noise into an image
* Output shape matches the dataset image size (28×28)

### Discriminator

* Takes an image as input (real or generated)
* Outputs a probability indicating whether the image is real or fake
* Trained as a binary classifier

---

## Training Methodology

* The GAN is trained using **manual optimization** in PyTorch Lightning
* Two optimizers are used:

  * One for the Generator
  * One for the Discriminator
* Training follows an alternating strategy:

  1. Train Discriminator on real images
  2. Train Discriminator on fake images
  3. Train Generator to fool the Discriminator

Loss functions used:

* Binary Cross-Entropy Loss for both Generator and Discriminator

---

## Outputs

### 1. Training Logs

Epoch-wise logs are printed during training in the following format:

```
Epoch 10/50 | D_loss: 0.53 | G_loss: 1.24
```

### 2. Generated Samples

* Folder: `generated_samples/`
* Images saved every `save_interval` epochs
* Each image contains a 5×5 grid of generated samples
* Filenames follow the format: `epoch_05.png`, `epoch_10.png`, etc.

### 3. Final Generated Images

* Folder: `final_generated_images/`
* Contains 100 synthetic images generated after training completion

### 4. Label Prediction of Generated Images

* A pre-trained classifier is used to predict labels for the final 100 generated images
* The label distribution is analyzed to evaluate generation diversity

---

## Technologies Used

* Python
* PyTorch
* PyTorch Lightning
* Torchvision
* Matplotlib

---

## How to Run

1. Configure dataset and training parameters
2. Initialize Generator and Discriminator models
3. Train the GAN using PyTorch Lightning Trainer
4. Monitor training logs and generated images
5. Evaluate generated images using a pre-trained classifier

---

## Conclusion

This experiment demonstrates how GANs can be used to generate realistic synthetic images. The quality of generated samples improves over training epochs as the Generator and Discriminator learn through adversarial training. This lab provides hands-on understanding of GAN architecture, training dynamics, and evaluation techniques.

