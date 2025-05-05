import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pgmpy.inference import VariableElimination
from bayesian_network import BayesianNetworkModel
from model import HiddenMarkovModel
from tqdm import tqdm

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

def load_and_split_data(filepath, train_ratio=0.7):
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    return train_df, test_df


def train_bayesian_network(train_df):
    train_df['TimeOfDay'] = train_df['timestamp'].apply(lambda x: discretize_time_simple(pd.to_datetime(x, unit='ms').hour))
    train_df['TempRange'] = train_df['temperature'].apply(discretize_temp_simple)
    train_df['NextState'] = train_df['location'] + "_" + train_df['activity']

    bn_model = BayesianNetworkModel()
    bn_model.build_structure()

    categorical_cols = ['TimeOfDay', 'TempRange', 'NextState']
    for c in categorical_cols:
        train_df[c] = train_df[c].astype('category')

    from pgmpy.estimators import MaximumLikelihoodEstimator
    bn = bn_model.get_model()
    mle = MaximumLikelihoodEstimator(bn, train_df)
    estimated_cpds = mle.get_parameters()
    bn_model.set_cpds(estimated_cpds)

    return bn_model


def evaluate_pipeline(test_df, hmm, bn_model, vocab, obs_dict):
    inv_obs_dict = {i: obs for obs, i in obs_dict.items()}
    inference = VariableElimination(bn_model.get_model())

    y_true = []
    y_pred = []

    sequence_length = 10
    for idx in tqdm(range(0, len(test_df) - sequence_length)):
        seq_slice = test_df.iloc[idx: idx + sequence_length]
        if len(seq_slice) < sequence_length:
            continue

     
        sequence = [f"{row['activity']}@{row['location']}" for _, row in seq_slice.iterrows()]
        encoded_seq = np.array([obs_dict.get(o, 0) for o in sequence[:-1]]) 

        hmm_preds = hmm.predict_next_observation_modified(encoded_seq, top_k=3)

        current_time = discretize_time_simple(pd.to_datetime(seq_slice.iloc[-1]['timestamp'], unit='ms').hour)
        current_temp = discretize_temp_simple(seq_slice.iloc[-1]['temperature'])

        query_res = inference.query(
            variables=["NextState"],
            evidence={"TimeOfDay": current_time, "TempRange": current_temp}
        )

        bn_probs = {state_val: p_val for state_val, p_val in zip(query_res.state_names["NextState"], query_res.values)}

        hmm_distribution = {}
        for obs_str, prob in hmm_preds:
            loc, act = obs_str.split("@")
            key = f"{loc}_{act}"
            hmm_distribution[key] = prob

        combined_probs = {state_val: bn_probs.get(state_val, 0) * hmm_distribution.get(state_val, 0)
                          for state_val in bn_probs.keys()}

     
        norm_factor = sum(combined_probs.values())
        if norm_factor > 0:
            combined_probs = {k: v / norm_factor for k, v in combined_probs.items()}

        if combined_probs:
            predicted_state = max(combined_probs.items(), key=lambda x: x[1])[0]
        else:
            predicted_state = None 

        actual_state = f"{seq_slice.iloc[-1]['location']}_{seq_slice.iloc[-1]['activity']}"

        if predicted_state:
            y_true.append(actual_state)
            y_pred.append(predicted_state)

   
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n📊 Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")


if __name__ == "__main__":

    train_df, test_df = load_and_split_data("../data/hmm_observations_with_time_temp.json")

    bn_model = train_bayesian_network(train_df)

    hmm = HiddenMarkovModel.load("saved_models/hmm_smart_home_full.pkl")

    vocab = sorted(set(f"{row['activity']}@{row['location']}" for _, row in train_df.iterrows()))
    obs_dict = {obs: i for i, obs in enumerate(vocab)}

    evaluate_pipeline(test_df, hmm, bn_model, vocab, obs_dict)