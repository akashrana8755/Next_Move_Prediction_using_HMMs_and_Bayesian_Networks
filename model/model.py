import numpy as np
import pickle

class HiddenMarkovModel:
    def __init__(self, states, observations):
        self.states = states
        self.observations = observations
        self.n_states = len(states)
        self.n_observations = len(observations)

        self.start_prob = np.ones(self.n_states) / self.n_states
        self.trans_prob = np.random.rand(self.n_states, self.n_states)
        self.trans_prob /= self.trans_prob.sum(axis=1, keepdims=True)

        self.emit_prob = np.random.rand(self.n_states, self.n_observations)
        self.emit_prob /= self.emit_prob.sum(axis=1, keepdims=True)

    def forward(self, obs_seq):
        T = len(obs_seq)
        alpha = np.zeros((T, self.n_states))

        alpha[0] = self.start_prob * self.emit_prob[:, obs_seq[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.trans_prob[:, j]) * self.emit_prob[j, obs_seq[t]]

        return alpha

    def backward(self, obs_seq):
        T = len(obs_seq)
        beta = np.zeros((T, self.n_states))

        beta[-1] = 1

        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.trans_prob[i] * self.emit_prob[:, obs_seq[t+1]] * beta[t+1])
        return beta

    def train(self, sequences, max_iter=100, tol=1e-4):
        for _ in range(max_iter):
            A_new = np.zeros_like(self.trans_prob)
            B_new = np.zeros_like(self.emit_prob)
            pi_new = np.zeros_like(self.start_prob)

            for obs_seq in sequences:
                T = len(obs_seq)
                alpha = self.forward(obs_seq)
                beta = self.backward(obs_seq)

                gamma = (alpha * beta) / np.sum(alpha[-1])
                xi = np.zeros((T-1, self.n_states, self.n_states))

                for t in range(T-1):
                    denominator = np.sum(
                        alpha[t, :, None] * self.trans_prob * self.emit_prob[:, obs_seq[t+1]] * beta[t+1]
                    )
                    xi[t] = (alpha[t, :, None] * self.trans_prob * self.emit_prob[:, obs_seq[t+1]] * beta[t+1]) / denominator

                A_new += np.sum(xi, axis=0)
                B_new += np.sum(gamma[:, :, None] * (np.arange(self.n_observations) == obs_seq[:, None]), axis=0)
                pi_new += gamma[0]

            self.trans_prob = A_new / A_new.sum(axis=1, keepdims=True)
            self.emit_prob = B_new / B_new.sum(axis=1, keepdims=True)
            self.start_prob = pi_new / pi_new.sum()
        
    def predict_next_observation(self, obs_seq):
        state_seq = self.predict(obs_seq)
        last_state_name = state_seq[-1]
        last_state_index = self.states.index(last_state_name)

        next_obs_index = np.argmax(self.emit_prob[last_state_index])
        return self.observations[next_obs_index]
            
    def predict(self, obs_seq):
        T = len(obs_seq)
        delta = np.zeros((T, self.n_states)) 
        psi = np.zeros((T, self.n_states), dtype=int)  

        delta[0] = self.start_prob * self.emit_prob[:, obs_seq[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                max_prob = np.max(delta[t-1] * self.trans_prob[:, j])
                delta[t, j] = max_prob * self.emit_prob[j, obs_seq[t]]
                psi[t, j] = np.argmax(delta[t-1] * self.trans_prob[:, j])

        states_seq = np.zeros(T, dtype=int)
        states_seq[-1] = np.argmax(delta[-1]) 
        for t in range(T-2, -1, -1):
            states_seq[t] = psi[t+1, states_seq[t+1]]

        return [self.states[i] for i in states_seq]

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def predict_next_observation_modified(self, obs_seq, top_k=3):
        state_seq = self.predict(obs_seq)
        last_state_name = state_seq[-1]
        last_state_index = self.states.index(last_state_name)

        probs = self.emit_prob[last_state_index]

        top_indices = np.argsort(probs)[::-1][:top_k]

        return [(self.observations[i], round(probs[i], 4)) for i in top_indices]

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)