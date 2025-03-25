import pandas as pd
import matplotlib.pyplot as plt
from model import HiddenMarkovModel

df = pd.read_csv("../data/sensor_data.csv")

hmm = HiddenMarkovModel.load("saved_models/hmm_smart_home.pkl")

latest_sequence = df.tail(5)['sensor'].tolist()
obs_dict = {"Motion Detected": 0, "No Motion": 1, "Sitting": 2, "Cooking": 3}
test_data = [obs_dict[o] for o in latest_sequence]
predicted_states = hmm.predict(test_data)

df['timestamp'] = pd.to_datetime(df['timestamp'])


plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['sensor'].map(obs_dict), marker='o', linestyle='-', label="Observed Sensor Data")
plt.xticks(rotation=45)
plt.xlabel("Timestamp")
plt.ylabel("Sensor Readings")
plt.title("Smart Home Sensor Data Over Time")
plt.legend()
plt.show()

print("Predicted Next States:", predicted_states)