# Mac Pro Visual Inspection using Anomalib Patchcore
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project implements a visual inspection system using the [Anomalib](https://github.com/openvinotoolkit/anomalib) Patchcore model to detect anomalies on high-resolution board images.

## 🔍 Overview

The system is designed for automated defect detection on Mac Pro PCB images captured from multiple angles. It trains a Patchcore model on clean ("good") samples and flags anomalies based on reconstruction and memory-based deviation scores.

## 🧠 Features

- ⚙️ Patchcore model patched to support:
  - Missing ground truth masks
  - CPU-only training
  - Safe `predict_step` fallback logic
- 🧼 Custom image preprocessing with `ToTensor` and normalization
- 📦 Handles structured MVTec-like folder data:
dataset/
transistor/
train/good/
test/good/
test/anomalous/
ground_truth/anomalous/

## 📁 File Structure

- `main_patchcore.py`: Main script with patched trainer/model/dataset logic.
- `requirements.txt`: Dependencies (install with `pip install -r requirements.txt`)
- `patchcore_model.pth`: (Optional) Saved model weights after training.

## 🛠 Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install anomalib and other dependencies
pip install -r requirements.txt
