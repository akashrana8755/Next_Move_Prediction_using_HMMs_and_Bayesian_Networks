import json
import pandas as pd
import numpy as np

from pgmpy.estimators import MaximumLikelihoodEstimator
from bayesian_network import BayesianNetworkModel

def discretize_time(timestamp_ms):
    hour = pd.to_datetime(timestamp_ms, unit='ms').hour
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

def discretize_temperature(temp):
    if temp < 18:
        return "Cold"
    elif temp < 25:
        return "Comfortable"
    else:
        return "Hot"

def train_bn_model(input_json="../data/hmm_observations_with_time_temp.json", 
                   output_model_path="saved_models/bn_model.pkl"):
    records = []
    with open(input_json, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    df['TimeOfDay'] = df['timestamp'].apply(discretize_time)
    df['TempRange'] = df['temperature'].apply(discretize_temperature)

    df['NextState'] = df['location'] + "_" + df['activity']

    bn_model = BayesianNetworkModel()
    bn_model.build_structure()

    categorical_cols = ['TimeOfDay', 'TempRange', 'NextState']
    for c in categorical_cols:
        df[c] = df[c].astype('category')

    from pgmpy.estimators import ParameterEstimator
    from pgmpy.estimators import BayesianEstimator
    
    bn = bn_model.get_model()
    
    mle = MaximumLikelihoodEstimator(bn, df)
    estimated_cpds = mle.get_parameters() 

    bn_model.set_cpds(estimated_cpds)

    bn_model.save_model(filepath=output_model_path)
    print(f"[INFO] BN model trained and saved to {output_model_path}")

if __name__ == "__main__":
    train_bn_model()