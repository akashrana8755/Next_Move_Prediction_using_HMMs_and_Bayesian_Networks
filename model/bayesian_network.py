import os
import joblib
from pgmpy.models import DiscreteBayesianNetwork 
from pgmpy.factors.discrete import TabularCPD

class BayesianNetworkModel:
    def __init__(self):
        self.bn = DiscreteBayesianNetwork()
        self.is_trained = False

    def build_structure(self):
        self.bn = DiscreteBayesianNetwork([
            ("TimeOfDay", "NextState"),
            ("TempRange", "NextState")
        ])

    def set_cpds(self, cpds):
        self.bn.add_cpds(*cpds)
        if not self.bn.check_model():
            raise ValueError("Invalid CPD configuration.")
        self.is_trained = True

    def save_model(self, filepath="bn_model.pkl"):
        with open(filepath, "wb") as f:
            joblib.dump(self.bn, f)

    def load_model(self, filepath="bn_model.pkl"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.bn = joblib.load(filepath)
        self.is_trained = True

    def get_model(self):
        return self.bn