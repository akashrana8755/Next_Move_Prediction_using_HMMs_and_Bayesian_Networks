import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pickle

class BayesianPredictionModel:
    def __init__(self):
        """
        Define the Bayesian Network Structure
        """
        self.model = BayesianNetwork([
            ("Time", "Activity"),  # Time influences Activity
            ("Activity", "NextLocation"),  # Activity influences Next Location
            ("HMM_Prediction", "Final_Prediction"),  # HMM influences Final Prediction
            ("Time", "Final_Prediction"),  # Time influences Final Prediction
            ("NextLocation", "Final_Prediction")  # Next Location influences Final Prediction
        ])

    def train(self, df):
        """
        Train the Bayesian Network using Maximum Likelihood Estimation (MLE)
        """
        estimator = MaximumLikelihoodEstimator(self.model, df)
        for node in self.model.nodes():
            self.model.fit(estimator, node)

    def predict(self, evidence):
        """
        Perform inference given some evidence (inputs)
        """
        inference = VariableElimination(self.model)
        result = inference.map_query(variables=["Final_Prediction"], evidence=evidence)
        return result["Final_Prediction"]

    def save(self, file_path):
        """
        Save Bayesian Network model.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        """
        Load Bayesian Network model.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)