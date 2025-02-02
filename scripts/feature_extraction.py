# Import necessary modules
from utils import get_vector
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import scipy.sparse
import config

def extract_vector_features(parser_config):
    """
    Extracts vector features for a given parser configuration, including word vectors and other linguistic features.

    Args:
        parser_config (dict): The parser configuration containing the stack and input.

    Returns:
        tuple: A tuple containing two elements:
            - feats (dict): A dictionary of extracted features.
            - dense_feats (list): A list containing the combined feature vectors.
    """
    dense_feats = []
    feats = {}

    # Extract words and tags from the parser config
    inp = parser_config["input"]
    stack = parser_config["stack"]

    # Handle the input features (current, previous, and next words)
    i_idx_0, i_form_0, i_lem_0, i_pos_0 = inp[0][:4] if len(inp) > 0 else ("", "N/A", "N/A", "N/A")
    i_idx_1, i_form_1, i_lem_1, i_pos_1 = inp[1][:4] if len(inp) > 1 else ("", "N/A", "N/A", "N/A")
    i_idx_2, i_form_2, i_lem_2, i_pos_2 = inp[2][:4] if len(inp) > 2 else ("", "N/A", "N/A", "N/A")
    i_idx_3, i_form_3, i_lem_3, i_pos_3 = inp[3][:4] if len(inp) > 3 else ("", "N/A", "N/A", "N/A")

    s_idx_0, s_form_0, s_lem_0, s_pos_0 = stack[-1][:4] if len(stack) > 0 else ("", "N/A", "N/A", "N/A")
    s_idx_1, s_form_1, s_lem_1, s_pos_1 = stack[-2][:4] if len(stack) > 1 else ("", "N/A", "N/A", "N/A")

    # Add features for input and stack words
    feats["input_0_form"] = i_form_0
    feats["input_0_lem"] = i_lem_0
    feats["input_0_pos"] = i_pos_0
    feats["input_1_form"] = i_form_1
    feats["input_1_pos"] = i_pos_1
    feats["input_2_lem"] = i_lem_2
    feats["input_2_pos"] = i_pos_2
    feats["input_3_pos"] = i_pos_3
    feats["stack_0_form"] = s_form_0
    feats["stack_0_pos"] = s_pos_0
    feats["stack_1_pos"] = s_pos_1
    feats["stack_1_lem"] = s_lem_1

    # Get word embeddings for the current word, previous word, and next word
    input_0_vec = get_vector(i_form_0)
    input_1_vec = get_vector(i_form_1) if i_form_1 != "N/A" else np.zeros(config.NO_OF_DIMENSIONS)
    input_2_vec = get_vector(i_form_2) if i_form_2 != "N/A" else np.zeros(config.NO_OF_DIMENSIONS)
    input_3_vec = get_vector(i_form_3) if i_form_3 != "N/A" else np.zeros(config.NO_OF_DIMENSIONS)
    stack_0_vec = get_vector(s_form_0) if s_form_0 != "N/A" else np.zeros(config.NO_OF_DIMENSIONS)
    stack_1_vec = get_vector(s_form_1) if s_form_1 != "N/A" else np.zeros(config.NO_OF_DIMENSIONS)

    # Combine all word vectors into a single feature vector (concatenate)
    combined_vector = np.hstack([stack_1_vec, stack_0_vec, input_3_vec, input_2_vec, input_1_vec, input_0_vec])

    # Append the combined vector to dense features
    dense_feats.append(combined_vector)

    # Return the resulting feature matrix as a dense array
    return feats, dense_feats

# Initialize lists to store raw features
train_features_raw = []
vector_features_raw = []

# Assuming train_configs is a list of parser configurations stored in config.TRAIN_CONFIGS
for parser_config in config.TRAIN_CONFIGS:
    # Extracting sparse and dense features using extract_vector_features function
    sparse_feats, dense_feats = extract_vector_features(parser_config)

    # Append the extracted features to the respective lists
    train_features_raw.append(sparse_feats)
    vector_features_raw.append(dense_feats)

# Vectorizing sparse features using DictVectorizer
vectoriser = DictVectorizer()
train_features = vectoriser.fit_transform(train_features_raw)

# Print the shape of the sparse feature matrix
print("Sparse feature shape:", train_features.shape)

# Stacking dense vector features into a matrix
vector_feats = np.vstack(vector_features_raw)

# Print the shape of the dense feature matrix
print("Dense feature shape:", vector_feats.shape)

# Combine vector and label features into a sparse matrix
config.COMBINED_FEATURES = scipy.sparse.hstack([train_features, vector_feats])
print("Combined feature shape:", config.COMBINED_FEATURES.shape)