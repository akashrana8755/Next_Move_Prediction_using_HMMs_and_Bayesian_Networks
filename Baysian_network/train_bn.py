import pandas as pd
from bayesian_network import BayesianPredictionModel
import os

# Load data
df = pd.read_csv("../data/sensor_data.csv")

# Convert categorical data to numerical encoding
time_mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
activity_mapping = {"Sitting": 0, "Cooking": 1, "Sleeping": 2, "No Motion": 3}
location_mapping = {"Living Room": 0, "Kitchen": 1, "Bedroom": 2, "Bathroom": 3}

df['Time'] = df['timestamp'].apply(lambda x: "Morning" if "08:00" in x else "Night")
df['Time'] = df['Time'].map(time_mapping)
df['Activity'] = df['activity'].map(activity_mapping)
df['NextLocation'] = df['location'].map(location_mapping)

# Initialize and train the Bayesian Network
bn = BayesianPredictionModel()
bn.train(df)

# Save the trained model
os.makedirs("saved_models", exist_ok=True)
bn.save("saved_models/bayesian_smart_home.pkl")

print("Bayesian Network trained and saved successfully!")