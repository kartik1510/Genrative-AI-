 Genrative-AI-
Week 1 lab

 Lab 1 – Image Generation Using Diffusion Model

 Course
CSET419 – Introduction to Generative AI  

 Experiment Title
Synthetic Image Data Generation Using a Pre-trained Diffusion Model

---

 Objective
The objective of this experiment is to generate synthetic image data using a pre-trained diffusion model. A Stable Diffusion model from the Hugging Face model hub is used to generate images based on textual prompts and store them as a dataset.

---

 Description
Generative AI models are capable of creating new data samples that resemble real-world data. In this experiment, a pre-trained Stable Diffusion model is loaded from Hugging Face using the Diffusers library. A list of text prompts is provided as input, and images are generated one by one for each prompt.  
Each generated image is saved to disk, and a mapping between prompts and image files is stored for reference.

---

 Model Used
- **Model Name:** Stable Diffusion v1.5  
- **Source:** Hugging Face Model Hub  
- **Model ID:** `runwayml/stable-diffusion-v1-5`

---

 Technologies Used
- Python
- PyTorch
- Hugging Face Diffusers
- Transformers
- Pillow (PIL)

---

 Setup Instructions

1. Install Required Libraries
```bash
pip install torch torchvision diffusers transformers accelerate pillow
