Overview

This lab focuses on understanding and implementing a Variational Autoencoder (VAE), a generative deep learning model capable of learning latent probability distributions and generating new data samples.
The experiment is performed using image data to study reconstruction quality, latent space behavior, and sample generation.

Unlike standard autoencoders, a VAE learns a distribution over the latent space, enabling it to generate novel and diverse outputs.

Objectives

Understand the working principle of Variational Autoencoders

Learn the difference between Autoencoders and VAEs

Implement a VAE using a deep learning framework

Train the VAE on image data

Generate new images from the learned latent space

Dataset

The experiment uses one of the following datasets:

MNIST (handwritten digits)

Fashion-MNIST (clothing items)

The dataset is normalized and split into training and testing sets before model training.

Model Architecture
Encoder

Takes an input image

Outputs:

Mean (μ)

Log variance (log σ²)

These parameters define a probability distribution in latent space.

Reparameterization Trick

To enable backpropagation through random sampling:

z = μ + σ × ε , where ε ~ N(0,1)

Decoder

Takes latent vector z

Reconstructs the input image

Loss Function

The VAE loss consists of two components:

Reconstruction Loss

Measures similarity between input and reconstructed image

Binary Cross-Entropy or Mean Squared Error

KL Divergence Loss

Regularizes latent space to follow a standard normal distribution

Total Loss = Reconstruction Loss + KL Divergence Loss

Training

The model is trained for multiple epochs

Training loss is monitored to observe convergence

Reconstructed images improve gradually during training

Sample Generation

After training:

Random latent vectors are sampled from a standard normal distribution

These vectors are passed through the decoder

The decoder generates new images that were not present in the training dataset

Results

The following outputs are obtained:

Trained VAE model

Reconstructed images

Newly generated images

Loss curve showing training progress

Technologies Used

Python

PyTorch / TensorFlow

NumPy

Matplotlib

Conclusion

This lab demonstrates how Variational Autoencoders learn smooth latent representations and generate new data samples. VAEs form the foundation for many modern generative models and help in understanding probabilistic deep learning concepts.
