# Generative Pneumonia X-ray Modeling  
**VAE-Based Pneumonia X-ray Generation on Google Cloud Platform (GCP)**  

## Overview  
This project explores **generative modeling of pneumonia X-rays** using a **Variational Autoencoder (VAE)**. The goal is to generate synthetic pneumonia-affected chest X-rays from patients’ baseline healthy scans, enabling data augmentation and improved generalization for diagnostic AI models.  

Developed and executed on **Google Cloud Platform (GCP)**, the project integrates cloud-based preprocessing, metadata management, and scalable architecture for handling large medical imaging datasets.

---

## Key Features  
- **VAE-Based Image Generation:**  
  Designed a deep Variational Autoencoder to model transformations from healthy to pneumonia-affected lung images.  

- **Custom Preprocessing Pipeline:**  
  Implemented with **OpenCV** and **NumPy** for normalization, denoising, and temporal alignment of X-ray sequences.  

- **Cloud Integration:**  
  Used **Cloud SQL** for structured metadata querying and orchestrated the data pipeline through **GCP Compute Engine**.  

- **Scalable Architecture:**  
  Modular design supports distributed execution and dataset versioning across cloud instances.  

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
Due to **limited computational resources**, full-scale model training and evaluation have not yet been completed.  
All components for preprocessing, orchestration, and model definition are fully implemented and ready for deployment on GPU/TPU-backed environments.  

---

