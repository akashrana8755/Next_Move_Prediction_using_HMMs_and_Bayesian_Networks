import numpy as np
import pandas as pd
from model import HiddenMarkovModel
import os
import json
from tqdm import tqdm  # progress bar

# Load JSON dataset
with open("../data/hmm_observations_100k.json", "r") as f:
    data = [json.loads(line) for line in f]

# Combine activity and location into unique observation tokens
observation_tokens = [f"{d['activity']}@{d['location']}" for d in data]

# Split into sequences of equal length (e.g. user sessions)
sequence_length = 10
sequences = [observation_tokens[i:i + sequence_length] for i in range(0, len(observation_tokens), sequence_length)]
sequences = [seq for seq in sequences if len(seq) == sequence_length]

# Build observation vocabulary
unique_obs = sorted(set(token for seq in sequences for token in seq))
obs_dict = {obs: i for i, obs in enumerate(unique_obs)}

# Encode observations
train_data = [np.array([obs_dict[o] for o in seq]) for seq in sequences]

# Define hidden states
states = [f"State{i}" for i in range(10)]

# Initialize HMM
hmm = HiddenMarkovModel(states=states, observations=unique_obs)

# Simulated verbose training: breaking into mini-batches
batch_size = 1000
num_batches = len(train_data) // batch_size

print("🔁 Starting training...")
for i in tqdm(range(num_batches), desc="Training Batches"):
    batch = train_data[i * batch_size:(i + 1) * batch_size]
    hmm.train(batch)  # Assumes train() can be called iteratively; if not, call once and add internal progress inside the model
    print(f"✅ Finished batch {i + 1}/{num_batches}")

print("✅ Training complete!")

# Save the model
os.makedirs("saved_models", exist_ok=True)
hmm.save("saved_models/hmm_smart_home_full.pkl")

print("💾 HMM model saved to 'saved_models/hmm_smart_home_full.pkl'")