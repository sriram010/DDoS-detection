# DDoS Attack Detection for IoT using Hybrid Deep Learning

This project implements a hybrid deep learning model combining CNN and LSTM for detecting DDoS attacks in IoT networks.

## Features

- Hybrid architecture combining CNN and LSTM networks
- Real-time DDoS attack detection
- Data preprocessing and sequence preparation
- Model evaluation with detailed metrics
- Visualization of results

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in CSV format with the following structure:
   - Features columns: network traffic features
   - Label column: 'label' (0 for normal traffic, 1 for DDoS attack)

2. Run the detection system:
```bash
python ddos_detection.py
```

## Model Architecture

The hybrid model consists of:
- CNN layers for spatial feature extraction
- LSTM layers for temporal dependency learning
- Dense layers for final classification
- Dropout layers for preventing overfitting

## Evaluation Metrics

The system provides:
- Classification report (Precision, Recall, F1-score)
- Confusion matrix
- Training history plots

## Note

Make sure to modify the `load_and_prepare_data` function according to your specific dataset structure.
