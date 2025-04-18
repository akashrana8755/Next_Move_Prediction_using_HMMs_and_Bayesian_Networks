import numpy as np
import json
from model import HiddenMarkovModel

# Load the trained model
model_path = "saved_models/hmm_smart_home_full.pkl"
hmm = HiddenMarkovModel.load(model_path)

# Prepare test sequence (activity@location format)
test_sequence = [
    "watching@bedroom1",
    "exercise@garden",
    "toilet@bathroom",
    "watching@bedroom1",
    "sleeping@bedroom1"
    
]

# Use the same dictionary used  in training
with open("../data/hmm_observations_100k.json", "r") as f:
    data = [json.loads(line) for line in f]
vocab = sorted(set(f"{d['activity']}@{d['location']}" for d in data))
obs_dict = {obs: i for i, obs in enumerate(vocab)}
inv_obs_dict = {i: obs for obs, i in obs_dict.items()}

# Encode test sequence
encoded_test_seq = np.array([obs_dict[o] for o in test_sequence])

# Predict most likely state sequence
predicted_states = hmm.predict(encoded_test_seq)
print("🔎 Predicted Hidden States:", predicted_states)

# Predict next likely observation
next_obs = hmm.predict_next_observation(encoded_test_seq)
activity, location = next_obs.split("@")

print(f"🔮 Predicted Next Activity: {activity}")
print(f"📍 Predicted Next Location: {location}")