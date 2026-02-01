# Variational Autoencoder (VAE) – Lab 3

Implementation of a **Variational Autoencoder (VAE)** for image reconstruction and generation as part of the *CSET419 – Introduction to Generative AI* course.

This repository demonstrates how VAEs learn probabilistic latent representations and generate new samples by modeling data distributions instead of fixed encodings.

---

## 📌 Features

* Variational Autoencoder implemented from scratch
* Encoder–decoder architecture with reparameterization trick
* KL Divergence + Reconstruction loss
* Image reconstruction and generation
* Training loss visualization

---

## 🧠 Concept Overview

A **Variational Autoencoder** differs from a standard autoencoder by learning a *distribution* over the latent space rather than a deterministic encoding.

The encoder outputs:

* Mean (μ)
* Log variance (log σ²)

A latent vector is sampled using:

```
z = μ + σ × ε   , ε ~ N(0,1)
```

This enables smooth latent spaces and meaningful data generation.

---

## 📂 Dataset

The project uses one of the following datasets:

* **MNIST** (handwritten digits)
* **Fashion-MNIST** (clothing items)

Preprocessing steps:

* Normalization of pixel values to range [0,1]
* Splitting into training and testing sets

---

## 🏗️ Model Architecture

### Encoder

* Input: Image
* Output: Mean (μ) and log variance (log σ²)

### Latent Space

* Sampling performed using the reparameterization trick

### Decoder

* Input: Latent vector `z`
* Output: Reconstructed image

---

## 📉 Loss Function

The total VAE loss consists of:

1. **Reconstruction Loss**

   * Binary Cross-Entropy or Mean Squared Error
   * Measures reconstruction quality

2. **KL Divergence Loss**

   * Regularizes latent space to follow a standard normal distribution

```
Total Loss = Reconstruction Loss + KL Divergence
```

---

## 🚀 Training

* Model trained for multiple epochs
* Training loss monitored during learning
* Reconstruction quality improves progressively

---

## 🎨 Sample Generation

After training:

* Latent vectors are sampled from a standard normal distribution
* Passed directly through the decoder
* New images are generated that were not part of the training data

---

## 📊 Results

The project produces:

* Trained VAE model
* Original vs reconstructed images
* Newly generated images
* Training loss curve

---

## 🛠️ Tech Stack

* Python
* PyTorch / TensorFlow
* NumPy
* Matplotlib

---

## 📌 Conclusion

This project demonstrates how Variational Autoencoders learn smooth latent representations and generate new data samples. VAEs form a foundational concept for advanced generative models such as diffusion models.
