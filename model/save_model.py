import pickle

def save_model(hmm, filename):
    with open(filename, "wb") as f:
        pickle.dump(hmm, f)

def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)