import pandas as pd
from  model import HiddenMarkovModel
from bayesian_network import BayesianPredictionModel

# Load trained models
hmm = HiddenMarkovModel.load("saved_models/hmm_smart_home.pkl")
bn = BayesianPredictionModel.load("saved_models/bayesian_smart_home.pkl")

# Load new sensor readings
df = pd.read_csv("../data/sensor_data.csv")
latest_sequence = df.tail(3)['sensor'].tolist()

# Convert to numeric format
obs_dict = {"Motion Detected": 0, "No Motion": 1, "Sitting": 2, "Cooking": 3}
test_data = [obs_dict[o] for o in latest_sequence]

# Step 1: HMM Prediction
hmm_predicted_states = hmm.predict(test_data)

# Step 2: Bayesian Network Refinement
latest_time = "Morning" if "08:00" in df.iloc[-1]['timestamp'] else "Night"
time_mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
latest_time = time_mapping[latest_time]

activity_mapping = {"Sitting": 0, "Cooking": 1, "Sleeping": 2, "No Motion": 3}
latest_activity = activity_mapping[df.iloc[-1]['activity']]

hmm_pred_location = {"Living Room": 0, "Kitchen": 1, "Bedroom": 2, "Bathroom": 3}[hmm_predicted_states[-1]]

# Perform Bayesian Inference
final_prediction = bn.predict({
    "Time": latest_time,
    "Activity": latest_activity,
    "HMM_Prediction": hmm_pred_location
})

# Convert back to readable format
location_reverse_mapping = {0: "Living Room", 1: "Kitchen", 2: "Bedroom", 3: "Bathroom"}
final_location = location_reverse_mapping[final_prediction]

print("HMM Predicted Location:", hmm_predicted_states[-1])
print("Refined Prediction by Bayesian Network:", final_location)