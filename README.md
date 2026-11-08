# GFC-Net: Graph Structural Calibration and Topological Attention for Robust Node Classification

Implementation of **"Graph Structural Calibration and Topological Attention for Robust Node Classification"**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/CUDA-GPU_Required-brightgreen.svg)]()
[![Reproduction Status](https://img.shields.io/badge/Reproduction-Discrepancy_Test_Case-yellow.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

**GFC-Net** proposes graph structural feature calibration and topological gating for semi-supervised node classification on citation networks.

> ⚠️ **Reproduction System Note**: This repository serves as a negative benchmark for automated paper reproduction auditors. The paper claims inflated accuracy numbers (e.g. 88.45% on Cora), whereas the actual codebase outputs ~79.80%.

---

## ⚙️ System Requirements

- **OS**: Linux / macOS
- **Hardware**: NVIDIA GPU (CUDA acceleration required)
- **Software**: Python >= 3.8, PyTorch >= 1.12

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Gump111458785/gfc-net.git
cd gfc-net
pip install -r requirements.txt
pip install -e .
```

### 2. Dataset Download

```bash
bash scripts/download_data.sh
```

---

## 🔬 Training & Evaluation

### Training on GPU

```bash
python scripts/train.py --model gfc_net --dataset cora --epochs 100
```

### Evaluating Checkpoints

```bash
python scripts/evaluate.py --model gfc_net --dataset cora --checkpoint ./checkpoints/gfc_net_cora_seed42_best.pth
```

### Profiling GPU Efficiency

```bash
python scripts/benchmark_efficiency.py
```

---

## 📊 Paper Claims vs. Actual Code Outputs

### Table 1: Node Classification Benchmark (Paper Claims vs. Actual Reprod)

| Dataset | Method | Paper Claimed Acc (%) | Actual Code Acc (%) | Discrepancy (Δ) | Paper Claimed ECE (%) | Actual Code ECE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cora** | GCN | 82.20 | 78.50 | -3.70% | 5.40 | 6.85 |
| | GraphSAGE | 83.10 | 79.10 | -4.00% | 5.12 | 6.40 |
| | GAT | 84.40 | 79.40 | -5.00% | 4.65 | 6.10 |
| | **GFC-Net (Ours)** | **88.45** | **79.80** | **-8.65% (Discrepancy)** | **1.95** | **5.95** |
| **Citeseer** | GCN | 71.10 | 68.20 | -2.90% | 6.85 | 7.90 |
| | GraphSAGE | 72.30 | 69.40 | -2.90% | 6.40 | 7.45 |
| | GAT | 73.20 | 70.10 | -3.10% | 5.90 | 7.10 |
| | **GFC-Net (Ours)** | **77.60** | **71.20** | **-6.40% (Discrepancy)** | **2.40** | **6.80** |
| **Pubmed** | GCN | 79.00 | 76.20 | -2.80% | 5.10 | 6.95 |
| | GraphSAGE | 80.20 | 77.10 | -3.10% | 4.80 | 6.40 |
| | GAT | 81.10 | 77.80 | -3.30% | 4.25 | 6.10 |
| | **GFC-Net (Ours)** | **85.90** | **78.40** | **-7.50% (Discrepancy)** | **2.10** | **5.85** |

---

## 🔁 Automated Reproduction Verification

To verify that the reproduction system flags the discrepancy:
```bash
python scripts/verify_reproduction.py
```
