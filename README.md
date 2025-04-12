# 🚀 HMM Smart Home Prediction

## Overview

This project implements a **Hidden Markov Model (HMM)** to predict human movement patterns in a smart home environment based on **location**, **activity**, and **time**.  
Additionally, it integrates a **Dynamic Bayesian Network (DBN)** to further refine predictions by combining probabilistic inferences from the HMM model.

## Project Structure

### HMM Model

| File | Description |
|------|-------------|
| `model.py` | Defines the HMM class and core algorithms (forward, backward, Viterbi, etc.). |
| `train.py` | Trains the HMM using the prepared dataset. |
| `test.py` | Tests the trained HMM and outputs the single most probable observation. |
| `test2.py` | Tests the trained HMM and outputs the top 3 most probable observations. |
| `evaluate.py` | Splits the dataset (70% train / 30% test), trains and evaluates the HMM. |

### DBN Model

| File | Description |
|------|-------------|
| `bayesian_network.py` | Defines the DBN class and inference algorithms. |
| `train_bn.py` | Trains the DBN using Maximum Likelihood Estimation (MLE). |

### Hybrid Model (HMM + DBN)

| File | Description |
|------|-------------|
| `test_bn.py` | Tests the trained DBN, taking HMM output as input, and predicts the most probable observation. |
| `evaluate_bn.py` | Splits the dataset, trains both HMM and DBN, and evaluates the overall hybrid system. |

### Utilities

| Folder / File | Description |
|---------------|-------------|
| `data/` | Contains training and testing datasets. |
| `saved_models/` | Stores the trained model files. |
| `save_model.py` | Utility for model saving and loading. |
| `utils.py` | Helper functions for data processing and utilities. |

---

## How to Run
1. Train the model:

# Train HMM model first 
python3 train.py

# Train DBN model
python3 train_bn.py

2. Test with new observation:

# Test the HMM model (for single output)
python3 test.py

# Test the HMM model (for 3 output)
python3 test2.py

# Test the hybrid model (HMM + DBN)
python3 test_bn.py

3. To Evaluate the model performance

# Evaluate HMM model performance
python3 evaluate.py

# Evaluate the performance of whole model
python3 evaluate_bn.py