import numpy as np
from model import HiddenMarkovModel
import os

states = ["Living Room", "Kitchen", "Bedroom", "Bathroom"]
observations = ["Motion Detected", "No Motion", "Sitting", "Cooking"]

hmm = HiddenMarkovModel(states, observations)

# Dummy Data for training
train_sequences = [
    ["Motion Detected", "Cooking", "No Motion", "Sitting"],
    ["Sitting", "Motion Detected", "No Motion", "Cooking"],
    ["No Motion", "Sitting", "Motion Detected", "Cooking"],
]

obs_dict = {obs: i for i, obs in enumerate(observations)}
train_data = [np.array([obs_dict[o] for o in seq]) for seq in train_sequences]

hmm.train(train_data)

os.makedirs("saved_models", exist_ok=True)
hmm.save("saved_models/hmm_smart_home.pkl")

print("HMM Model trained and saved successfully!")