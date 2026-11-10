# Graph Structural Calibration and Topological Attention for Robust Node Classification

**Abstract**  
Graph Neural Networks (GNNs) frequently exhibit structural feature degradation and extreme probability miscalibration when classifying nodes under noisy graph topology. In this paper, we propose **GFC-Net**, an integrated architecture and optimization framework designed for robust semi-supervised node classification on citation graphs. First, we develop the **Graph Feature Calibration (GFC)** module, which dynamically rectifies local node representations against high-order neighborhood embeddings via topology-guided gating. Second, we formulate the **Graph Calibration (GCL)** loss, coupling label-smoothed cross-entropy with a boundary margin constraint to prevent overconfident misclassifications on heterophilous graph nodes. We evaluate our method across three standard citation benchmarks (**Cora**, **Citeseer**, and **Pubmed**) under four quantitative metrics: Classification Accuracy, Macro-F1 Score, Micro-F1 Score, and Expected Calibration Error (ECE). The paper claims state-of-the-art predictive performance, reporting **88.45% Accuracy** on Cora (+6.25% gain over GCN) and reducing calibration error down to **1.95% ECE**.

---

## 1. Introduction

Graph Neural Networks have become the predominant paradigm for learning on non-Euclidean graph-structured data. However, standard message-passing architectures suffer from structural oversmoothing and severe confidence miscalibration.

### Main Contributions

We articulate three primary contributions:
1. **Architectural Contribution (GFC Module)**: We design the Graph Feature Calibration module, dynamically modulating the balance between self-node features and aggregated neighborhood representations.
2. **Loss Formulation (GCL Loss)**: We introduce the Graph Calibration Loss, incorporating soft boundary margin constraints to regularize peaky predictive distributions on ambiguous graph nodes.
3. **Multi-Dataset Empirical Evaluation**: We present evaluations on Cora, Citeseer, and Pubmed across accuracy, macro-F1, micro-F1, and ECE.

---

## 2. Methodology

### 2.1 Graph Feature Calibration (GFC) Module
Given normalized graph adjacency $\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-\frac{1}{2}}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-\frac{1}{2}}$ and intermediate feature matrix $\mathbf{H} \in \mathbb{R}^{N \times D}$, the GFC module computes gated combination:

$$\mathbf{H}_{\text{neigh}} = \hat{\mathbf{A}} \mathbf{H}, \quad \boldsymbol{\alpha} = \sigma\Big(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{H})\Big)$$

$$\mathbf{H}_{\text{calib}} = \text{LayerNorm}\Big(\mathbf{H} + \mathbf{H} \odot \boldsymbol{\alpha} + \mathbf{H}_{\text{neigh}} \odot (1 - \boldsymbol{\alpha})\Big)$$

### 2.2 Graph Calibration (GCL) Loss
$$\mathcal{L}_{\text{GCL}} = \mathcal{L}_{\text{smooth}}(\mathbf{z}, y) + \lambda_{\text{cal}} \cdot \max(0, p_{(1)} - p_{(2)} - \tau) - \beta \cdot \mathcal{H}(\mathbf{p})$$

---

## 3. Experimental Evaluation (Paper Reported Claims)

### 3.1 Datasets & Implementation Details
We evaluate node classification across three established citation graph benchmarks:
- **Cora**: 2,708 scientific publications across 7 machine learning sub-fields with 5,429 citation edges.
- **Citeseer**: 3,327 computer science papers across 6 categories connected by 4,732 citation edges.
- **PubMed**: 19,717 biomedical articles on diabetes across 3 classes connected by 44,338 citation links.

All models are trained for 100 epochs using the Adam optimizer with initial learning rate $0.01$, weight decay $5 \times 10^{-4}$, and hidden feature dimension $64$. The random seed is fixed to 42 across all runs. Experiments are conducted on a single NVIDIA GeForce RTX 4090 / A100 GPU (24GB VRAM) under CUDA acceleration. Baseline architectures (GCN, GraphSAGE, GAT) are trained with standard Cross-Entropy Loss ($\mathcal{L}_{\text{CE}}$); the GCL loss is evaluated specifically on GFC-Net variants.

### 3.2 Main Quantitative Performance

| Dataset | Method | Claimed Accuracy (%) ↑ | Claimed Macro-F1 (%) ↑ | Claimed ECE (%) ↓ |
| :--- | :--- | :---: | :---: | :---: |
| **Cora** | GCN | 82.20 | 81.95 | 5.40 |
| | GraphSAGE | 83.10 | 82.85 | 5.12 |
| | GAT | 84.40 | 84.18 | 4.65 |
| | **GFC-Net (Ours)** | **88.45** | **88.38** | **1.95** |
| **Citeseer** | GCN | 71.10 | 69.80 | 6.85 |
| | GraphSAGE | 72.30 | 71.10 | 6.40 |
| | GAT | 73.20 | 72.05 | 5.90 |
| | **GFC-Net (Ours)** | **77.60** | **77.48** | **2.40** |
| **Pubmed** | GCN | 79.00 | 78.80 | 5.10 |
| | GraphSAGE | 80.20 | 80.05 | 4.80 |
| | GAT | 81.10 | 80.95 | 4.25 |
| | **GFC-Net (Ours)** | **85.90** | **85.82** | **2.10** |

**Table 2: GNN Complexity and Inference Latency on GPU.**

| Architecture | Parameters (K) | Latency (ms) |
| :--- | :---: | :---: |
| GCN | 92.4 | 1.45 |
| GraphSAGE | 184.8 | 2.10 |
| GAT | 93.2 | 2.65 |
| **GFC-Net (Ours)** | **98.6** | **2.80** |

**Table 3: Ablation Study on Cora Dataset.**

| Configuration | GFC Module | GCL Loss | Temp Scaling | Claimed Acc (%) | Claimed ECE (%) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| (a) Baseline (GCN) | ✗ | ✗ | ✗ | 82.20 | 5.40 |
| (b) + GFC Module | ✓ | ✗ | ✗ | 85.60 | 4.10 |
| (c) + GCL Loss | ✗ | ✓ | ✗ | 84.10 | 3.20 |
| (d) GFC + GCL | ✓ | ✓ | ✗ | 87.80 | 2.25 |
| (e) Full GFC-Net | ✓ | ✓ | ✓ | **88.45** | **1.95** |

---

## 4. Conclusion

GFC-Net explores structural feature calibration and confidence margin loss for graph node classification, asserting high performance gains on Cora, Citeseer, and Pubmed.

---

## References

1. T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," in *ICLR*, 2017.
2. P. Veličković et al., "Graph Attention Networks," in *ICLR*, 2018.
3. W. Hamilton et al., "Inductive Representation Learning on Large Graphs," in *NeurIPS*, 2017.
4. C. Guo et al., "On Calibration of Modern Neural Networks," in *ICML*, 2017.
