# Backdoor Attack against Log Anomaly Detection Models

This repository contains the code and instructions for the paper **"Backdoor Attack against Log Anomaly Detection Models"** by **He Cheng, Depeng Xu, and Shuhan Yuan**. The paper introduces a novel backdoor attack framework for self-supervised log anomaly detection models (DeepLog and LogBERT), enabling abnormal log entries to evade detection.

## Overview

Log anomaly detection is critical in various systems for identifying system faults and novel attacks. This project demonstrates how to inject backdoors into log anomaly detection models to make abnormal logs evade detection. We use two state-of-the-art models, DeepLog and LogBERT, to show the effectiveness of our attack approach.

## Requirements

- Python 3.8+
- PyTorch 1.8.1+
- Transformers library for LogBERT
- Additional Python libraries: `numpy`, `pandas`, `scikit-learn`

Install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

## Datasets

We evaluate our method on the **BlueGene/L (BGL)** and **Thunderbird** log datasets. The datasets can be downloaded from the following sources:
- [BlueGene/L (BGL)](https://github.com/logpai/loghub/tree/master/BGL)
- [Thunderbird](https://github.com/logpai/loghub/tree/master/Thunderbird)

## Results

The experimental results demonstrate that the backdoored DeepLog and LogBERT models achieve high attack success rates (ASR) while maintaining strong benign performance (BP). For more details on the results, please refer to the paper.

## Paper

For more information about the methodology and experimental setup, please refer to our paper:

- **Title**: Backdoor Attack against Log Anomaly Detection Models
- **Authors**: He Cheng, Depeng Xu, Shuhan Yuan
- **Abstract**: [Link to your paper or preprint]
