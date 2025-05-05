import pandas as pd
from pgmpy.inference import VariableElimination
from bayesian_network import BayesianNetworkModel
from model import HiddenMarkovModel
import numpy as np
import json

model_path = "saved_models/hmm_smart_home_full.pkl"
hmm = HiddenMarkovModel.load(model_path)

with open("../data/hmm_observations_100k.json", "r") as f:
    data = [json.loads(line) for line in f]

vocab = sorted(set(f"{d['activity']}@{d['location']}" for d in data))
obs_dict = {obs: i for i, obs in enumerate(vocab)}
inv_obs_dict = {i: obs for obs, i in obs_dict.items()}


test_sequence = [
    "watching@bedroom1",
    "toilet@bathroom",
    "watching@bedroom1",
    "sleeping@bedroom1",
    "watching@bedroom1",
]

encoded_test_seq = np.array([obs_dict[o] for o in test_sequence])

predicted_states = hmm.predict(encoded_test_seq)
print("🔎 Predicted Hidden States:", predicted_states)

hmm_preds = hmm.predict_next_observation_modified(encoded_test_seq, top_k=3)
print("\n[INFO] HMM top-3 predicted next observations:")
for obs_tuple, prob in hmm_preds:
    print(f"  {obs_tuple} -> {prob:.4f}")


def test_bn_model(bn_model_path="saved_models/bn_model.pkl"):
    bn_model = BayesianNetworkModel()
    bn_model.load_model(filepath=bn_model_path)
    bn = bn_model.get_model()

    def discretize_time_simple(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 22:
            return "Evening"
        else:
            return "Night"

    def discretize_temp_simple(temp):
        if temp < 18:
            return "Cold"
        elif temp < 25:
            return "Comfortable"
        else:
            return "Hot"

    current_hour = 18     
    current_temp = 19.2   
    current_time_of_day = discretize_time_simple(current_hour)
    current_temp_range = discretize_temp_simple(current_temp)

    hmm_distribution = {}
    for obs_str, prob in hmm_preds:
        loc, act = obs_str.split("@")
        key = f"{loc}_{act}"
        hmm_distribution[key] = prob

    inference = VariableElimination(bn)
    query_res = inference.query(
        variables=["NextState"],
        evidence={"TimeOfDay": current_time_of_day, "TempRange": current_temp_range}
    )

    bn_probs = {}
    for state_val, p_val in zip(query_res.state_names["NextState"], query_res.values):
        bn_probs[state_val] = p_val

    combined_probs = {}
    for state_val in bn_probs.keys():
        hmm_prob = hmm_distribution.get(state_val, 0.0)
        combined_probs[state_val] = bn_probs[state_val] * hmm_prob

    norm_factor = sum(combined_probs.values())
    if norm_factor > 0:
        for k in combined_probs:
            combined_probs[k] /= norm_factor

    print("\n[INFO] BN distribution for NextState (given TimeOfDay & TempRange):")
    for s, p in bn_probs.items():
        print(f"  {s}: {p:.4f}")

    print("\n[INFO] HMM distribution for NextState (top-3):")
    for s, p in hmm_distribution.items():
        print(f"  {s}: {p:.4f}")

    print("\n[INFO] Final fused distribution:")
    sorted_combined = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)
    for s, p in sorted_combined:
        if p > 0:
            print(f"  {s}: {p:.4f}")

    if sorted_combined and sorted_combined[0][1] > 0:
        best_state, best_prob = sorted_combined[0]
        print(f"\n[INFO] Most probable next move => {best_state} (prob={best_prob:.4f})")
    else:
        print("\n[INFO] Could not determine a most probable next move (all zero).")

if __name__ == "__main__":
    test_bn_model("saved_models/bn_model.pkl")