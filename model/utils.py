def normalize_matrix(matrix):
    return matrix / matrix.sum(axis=1, keepdims=True)

def sequence_to_numeric(sequence, mapping):
    return [mapping[item] for item in sequence]