# Unknown-aware Dual-temperature Open-set Contrastive Graph Domain Adaptation (UCGA)
This repository contains the author's implementation in PyTorch for the paper "Unknown-aware Dual-temperature Open-set Contrastive Graph Domain Adaptation".
# Environment Requirement
The experiments were conducted on a single NVIDIA RTX 3090 GPU with 24GB memory. The required packages are as follows:
- python==3.8.17
- torch==1.7.1
- numpy==1.19.5
- scipy==1.10.1
- scikit-learn==1.3.0
- torch-geometric==2.1.0
# Datasets
data/ contains the 8 datasets used in our paper, i.e., Citation-v1, DBLP-v4, Amazon Computers 1, Amazon Computers 2, LastFM Asia 1, LastFM Asia 2, OGBN-Arxiv 1 and OGBN-Arxiv 2.

Each ".mat" file stores a network dataset, where

the variable "adjacency_matrix" represents an adjacency matrix,

the variable "features" represents a node attribute matrix,

the variable "labels" represents a node label matrix.
# Code
"models.py" is the UCGA model.

"train.py" is the open-set cross-network node classification over all 6 tasks.
# Please cite our paper as:
Zhihao Chen, Xiao Shen, Xiao Wang, Qingfeng Wang, and Xi Zhou, "Unknown-aware Dual-temperature Open-set Contrastive Graph Domain Adaptation", IEEE Trans. Pattern Anal. Mach. Intell., 2026.