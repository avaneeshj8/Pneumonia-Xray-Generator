# Generative Pneumonia X-ray Modeling  
**VAE-Based Pneumonia X-ray Generation on Google Cloud Platform (GCP)**  

## Overview  
This project explores **generative modeling of pneumonia X-rays** using a **Variational Autoencoder (VAE)** framework. The objective is to generate synthetic pneumonia-affected chest X-rays from patients’ baseline healthy scans to support data augmentation and improve diagnostic AI generalization.  

Developed and deployed on **Google Cloud Platform (GCP)**, this project integrates cloud-based preprocessing, structured metadata management, and scalable infrastructure for handling large medical imaging datasets.

---

## Key Features  
- **VAE-Based Image Generation:**  
  Designed a deep Variational Autoencoder to model transformations from healthy to pneumonia-affected lung images.  

- **Custom Preprocessing Pipeline:**  
  Built using **OpenCV** and **NumPy** for normalization, denoising, and temporal alignment of X-ray sequences.  

- **Cloud Integration:**  
  Leveraged **Cloud SQL** for structured metadata queries and managed pipeline orchestration through **GCP Compute Engine**.  

- **Modular, Scalable Design:**  
  Supports distributed execution, dataset versioning, and scalable deployment across multiple cloud instances.  

---
## Dataset  

This project utilizes the **Chest X-Ray Images (Pneumonia)** dataset, one of the most widely available open medical imaging datasets on **Kaggle**.  

- **Source:** [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **Description:** The dataset contains chest X-ray images categorized as **Normal** or **Pneumonia**. Each image is a posterior-anterior (PA) chest scan, collected from pediatric patients aged 1–5 years.  
- **Usage in this project:**  
  - Healthy X-rays serve as the **input domain** for the VAE.  
  - Pneumonia X-rays represent the **target domain** for generative modeling.  
  - Preprocessing includes grayscale normalization, denoising, resizing to a consistent spatial resolution, and temporal alignment where applicable.  

The dataset provides a robust and ethically accessible foundation for generative modeling in medical imaging research.

---

## Model Architecture  

The **Generative Pneumonia X-ray Model** is based on a **Conditional Variational Autoencoder (CVAE)** integrated with a **U-Net backbone**, enhanced using **adversarial (GAN) training** and **perceptual loss** for high-fidelity image generation.  

### Core Components  

#### 1. Encoder (U-Net Encoder)
- Multi-scale **U-Net** structure for spatial feature extraction.  
- Generates skip connections for preserving fine anatomical details.  
- Outputs mean (`μ`) and log-variance (`logσ²`) for the latent space representation.  

#### 2. Latent Mapper
- Applies the **reparameterization trick** to sample latent vector `z` from the learned distribution.  
- Provides a compressed, disease-conditioned latent representation for decoding.  

#### 3. Decoder (U-Net Decoder with FiLM Conditioning)
- Reconstructs pneumonia-affected X-rays from latent vector `z`.  
- Uses **FiLM (Feature-wise Linear Modulation)** for label-based conditioning.  
- Combines encoder skip connections to preserve spatial precision.  

#### 4. PatchGAN Discriminator
- Employs a **PatchGAN** discriminator for localized realism.  
- Operates on image patches to ensure texture and structural consistency.  
- Conditions on disease labels to improve fidelity and feature alignment.  

#### 5. Perceptual Feature Extractor
- Utilizes pretrained **DenseNet121** (via `torchxrayvision`) or **VGG16** for perceptual feature loss.  
- Compares feature-level similarity between generated and real X-rays to maintain anatomical realism.  

---

### Objective Functions  

| Loss Term | Description | Weight |
|------------|--------------|---------|
| **Adversarial Loss (L<sub>adv</sub>)** | Encourages realistic pneumonia generation via PatchGAN. | λ<sub>adv</sub> = 0.5 |
| **Perceptual Loss (L<sub>perc</sub>)** | Preserves high-level structural features. | λ<sub>perc</sub> = 1.0 |
| **Reconstruction Loss (L<sub>1</sub>)** | Minimizes pixel-wise differences between generated and real X-rays. | λ<sub>L1</sub> = 10.0 |
| **KL Divergence (L<sub>KL</sub>)** | Regularizes latent space to approximate a Gaussian prior. | λ<sub>KL</sub> = 1.0 |

**Total Generator Loss:**  
`L_G = λ_adv * L_adv + λ_perc * L_perc + λ_L1 * L_L1 + λ_KL * L_KL`


---

### Summary  
The final architecture — **CVAE + U-Net + PatchGAN + Perceptual Loss** — provides a robust framework for generating anatomically consistent pneumonia X-rays conditioned on healthy baselines, bridging generative modeling and medical imaging synthesis.  

---

## Tech Stack  
| Component | Technology |
|------------|-------------|
| **Cloud Infrastructure** | Google Cloud Platform (GCP) |
| **Deep Learning Framework** | TensorFlow / PyTorch |
| **Image Processing** | OpenCV, NumPy |
| **Database** | Cloud SQL |
| **Data Storage** | Google Cloud Storage |
| **Version Control** | Git, GitHub |

---

## Current Status  
Due to **limited computational resources**, full-scale model training and validation have not yet been completed.  
However, all components for preprocessing, orchestration, and model definition are fully implemented and ready for deployment on GPU or TPU-backed environments.  
