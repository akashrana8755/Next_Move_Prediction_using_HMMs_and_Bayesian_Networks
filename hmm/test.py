from model import HiddenMarkovModel

hmm = HiddenMarkovModel.load("saved_models/hmm_smart_home.pkl")

test_sequence = ["Motion Detected", "Cooking", "No Motion"]

obs_dict = {"Motion Detected": 0, "No Motion": 1, "Sitting": 2, "Cooking": 3}
test_data = [obs_dict[o] for o in test_sequence]

predicted_states = hmm.predict(test_data)

print("Predicted State Sequence:", predicted_states)