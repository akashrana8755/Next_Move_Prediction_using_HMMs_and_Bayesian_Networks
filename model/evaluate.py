import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from model import HiddenMarkovModel
from sklearn.metrics import accuracy_score

# =========================
# 1. Load Dataset
# =========================
with open("../data/hmm_observations_100k.json", "r") as f:
    data = [json.loads(line) for line in f]

# Combine activity and location into unique observation tokens
observation_tokens = [f"{d['activity']}@{d['location']}" for d in data]

# Split into sequences of equal length (e.g., user sessions)
sequence_length = 10
sequences = [observation_tokens[i:i + sequence_length] for i in range(0, len(observation_tokens), sequence_length)]
sequences = [seq for seq in sequences if len(seq) == sequence_length]

# Build observation vocabulary
unique_obs = sorted(set(token for seq in sequences for token in seq))
obs_dict = {obs: i for i, obs in enumerate(unique_obs)}
inv_obs_dict = {i: obs for obs, i in obs_dict.items()}

# Encode sequences
encoded_sequences = [np.array([obs_dict[o] for o in seq]) for seq in sequences]

# =========================
# 2. Split into Train and Test
# =========================
split_idx = int(0.7 * len(encoded_sequences))
train_data = encoded_sequences[:split_idx]
test_data = encoded_sequences[split_idx:]

print(f"Train sequences: {len(train_data)}")
print(f"Test sequences: {len(test_data)}")

# =========================
# 3. Train the HMM
# =========================
states = [f"State{i}" for i in range(10)]
hmm = HiddenMarkovModel(states=states, observations=unique_obs)

print("🔁 Starting training...")
batch_size = 1000
num_batches = len(train_data) // batch_size

for i in tqdm(range(num_batches), desc="Training Batches"):
    batch = train_data[i * batch_size:(i + 1) * batch_size]
    hmm.train(batch)

print("✅ Training complete!")

# =========================
# 4. Evaluate on Test Data
# =========================
y_true = []
y_pred = []

print("🔍 Evaluating model...")
for seq in tqdm(test_data, desc="Testing Sequences"):
    input_seq = seq[:-1]  # use first 9 as input
    true_next_obs = seq[-1]  # last observation is ground truth

    try:
        predicted_next_obs_idx = hmm.predict_next_observation(input_seq)
        predicted_next_obs = obs_dict[predicted_next_obs_idx] if isinstance(predicted_next_obs_idx, str) else predicted_next_obs_idx
    except Exception as e:
        # If prediction fails (e.g., unseen state), skip
        continue

    y_true.append(true_next_obs)
    y_pred.append(predicted_next_obs)

# Convert indices back to tokens for human-readable accuracy
y_true_tokens = [inv_obs_dict[idx] for idx in y_true]
y_pred_tokens = [inv_obs_dict[idx] for idx in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_true_tokens, y_pred_tokens)
print(f"\n🎯 Next Observation Prediction Accuracy: {accuracy * 100:.2f}%")

# Optionally: Save evaluation results
os.makedirs("evaluation_results", exist_ok=True)
pd.DataFrame({
    "True Observation": y_true_tokens,
    "Predicted Observation": y_pred_tokens
}).to_csv("evaluation_results/hmm_evaluation.csv", index=False)

print("✅ Evaluation results saved to 'evaluation_results/hmm_evaluation.csv'")
