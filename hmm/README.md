# HMM Smart Home Prediction

## Overview
This project uses a Hidden Markov Model (HMM) to predict human movement in a smart home environment based on location, activity, and time.

## Project Structure
- `model.py`: Defines the HMM class and algorithms.
- `train.py`: Trains the HMM using data.
- `test.py`: Tests the trained HMM with new observations.
- `save_model.py`: Handles model saving and loading.
- `utils.py`: Helper functions.
- `data/`: Folder for storing training and testing data.
- `saved_models/`: Folder for saving trained models.

## How to Run
1. Train the model:

python train.py

2. Test with new observation:

python test.py

3. To visualize results

python visualize.py