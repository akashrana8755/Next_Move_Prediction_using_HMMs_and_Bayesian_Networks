# train.py

import json
import pandas as pd
import numpy as np

from pgmpy.estimators import MaximumLikelihoodEstimator
from bayesian_network import BayesianNetworkModel

def discretize_time(timestamp_ms):
    """
    Convert a timestamp (in ms) to a discrete time-of-day category.
    Very naive example: 4 categories by hour range.
    Adjust the hour boundaries or approach as needed.
    """
    # Example: Convert from ms to hour-of-day, ignoring date
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
    """------------------------------------------------------------------------------------------------------------------------------------
    Convert a continuous temperature to a discrete range.
    Adjust thresholds as needed.
    """
    if temp < 18:
        return "Cold"
    elif temp < 25:
        return "Comfortable"
    else:
        return "Hot"

def train_bn_model(input_json="../data/hmm_observations_with_time_temp.json", 
                   output_model_path="bn_model.pkl"):
    """
    Train a BN from occupant data (location, activity, time, temperature, etc.).
    Saves the final BN model to disk.
    """
    # 1) Load data from JSON lines
    records = []
    with open(input_json, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    # Expected columns: ['activity', 'location', 'timestamp', 'temperature']
    # We want to predict the occupant's "NextState" => (location, activity),
    # or at least some form of it. For training a BN, we might store 
    # the current "TimeOfDay", "TempRange", and the occupant's current or next state.

    # 2) Create columns for discrete time + temperature
    df['TimeOfDay'] = df['timestamp'].apply(discretize_time)
    df['TempRange'] = df['temperature'].apply(discretize_temperature)

    # 3) Create a single "NextState" combined from (location, activity).
    #    Depending on your approach, you might use the next row in the sequence
    #    as the "next state". For simplicity, let's just treat each row's
    #    location+activity as 'NextState'.
    #    In a real pipeline, you'd shift the sequence so that you have
    #    TimeOfDay(t), TempRange(t) => NextState(t+1).
    #    But here's a simplified demonstration:
    df['NextState'] = df['location'] + "_" + df['activity']

    # 4) Build the BN structure, then estimate CPDs from the data using MLE
    bn_model = BayesianNetworkModel()
    bn_model.build_structure()

    # Convert the DataFrame columns to categorical
    # (so pgmpy can handle them as discrete variables).
    # For example:
    categorical_cols = ['TimeOfDay', 'TempRange', 'NextState']
    for c in categorical_cols:
        df[c] = df[c].astype('category')

    # 5) Fit the BN using MaximumLikelihoodEstimator
    #    This automatically estimates CPDs for each node from the data.
    from pgmpy.estimators import ParameterEstimator
    from pgmpy.estimators import BayesianEstimator
    
    # If your BN is already structured as: TimeOfDay -> NextState, TempRange -> NextState,
    # we can do:
    bn = bn_model.get_model()
    
    # We'll do MLE or Bayesian Estimation. Let's do MLE:
    mle = MaximumLikelihoodEstimator(bn, df)
    estimated_cpds = mle.get_parameters()  # This returns a list of CPDs

    # 6) Add these CPDs to our BN model
    bn_model.set_cpds(estimated_cpds)

    # 7) Save the BN to disk
    bn_model.save_model(filepath=output_model_path)
    print(f"[INFO] BN model trained and saved to {output_model_path}")

if __name__ == "__main__":
    train_bn_model()