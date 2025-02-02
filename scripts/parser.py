import scipy.sparse
import numpy as np
from utils import get_vector
import config
from config import TRAINING_SENTENCES

def is_projective(sentence):
    """
    Checks if a dependency tree is projective, i.e., has no crossing arcs.

    Args:
        sentence (list of tuple): List of (dependent, head) pairs where each tuple
                                   represents a dependency relation in the sentence.

    Returns:
        bool: True if the tree is projective, False otherwise.
    """
    # Create dependency tree with adjusted indices (1-based indexing)
    dependency_tree = [(dep + 1, w[-1]) for dep, w in enumerate(sentence)]

    # Sort edges by the positions of x (dependent) and y (head)
    sorted_edges = sorted(dependency_tree, key=lambda edge: (min(edge), max(edge)))

    for i, (x1, y1) in enumerate(sorted_edges):
        for j, (x2, y2) in enumerate(sorted_edges):
            # Skip checking the same edge
            if i >= j:
                continue

            # Order edges consistently
            min1, max1 = sorted((x1, y1))
            min2, max2 = sorted((x2, y2))

            # Check for crossing
            if min1 < min2 < max1 < max2 or min2 < min1 < max2 < max1:
                return False

    return True


def simulate_parsing(training_sentences):
    """
    Simulates parsing of training sentences and returns configurations and operations.

    Args:
        training_sentences (list of list): List of sentences, where each sentence is
                                            represented as a list of (word, label, token, dep) tuples.

    Returns:
        tuple: A tuple containing two lists:
            - configs (list): List of configurations during parsing.
            - ops (list): List of operations (SHIFT, LEFT-ARC, RIGHT-ARC).
    """
    ops = []
    configs = []
    num_non_proj = 0

    for sentence in training_sentences:
        # Discard non-projective sentences
        if not is_projective(sentence):
            num_non_proj += 1
            continue

        stack = [tuple([0, "ROOT"] + ["_"] * (len(sentence[0]) - 2))]
        input = [tuple([i + 1] + list(x[:-1])) for i, x in enumerate(sentence)]
        deps = [int(d) for w, l, t, d in sentence]

        while input:
            top_stack = stack[-1] if stack else None
            top_input = input[0]

            if top_stack is None or (top_stack[0] == 0 and len(input) > 1):
                ops.append("SHIFT")
                configs.append({
                    "stack": list(stack),
                    "input": list(input)
                })
                x = input.pop(0)
                stack.append(x)
                continue

            top_stack_dep = deps[top_stack[0] - 1]
            top_input_dep = deps[top_input[0] - 1]

            if top_stack_dep == top_input[0] and top_stack[0] not in [deps[x[0] - 1] for x in input]:
                ops.append("LEFT-ARC")
                configs.append({
                    "stack": list(stack),
                    "input": list(input)
                })
                stack.pop()
            elif top_input_dep == top_stack[0] and top_input[0] not in [deps[x[0] - 1] for x in input[1:]]:
                ops.append("RIGHT-ARC")
                configs.append({
                    "stack": list(stack),
                    "input": list(input)
                })
                x = stack.pop()
                input[0] = x
            else:
                ops.append("SHIFT")
                configs.append({
                    "stack": list(stack),
                    "input": list(input)
                })
                x = input.pop(0)
                stack.append(x)

        assert len(configs[-1]["stack"]) == 0
        assert ops[-1] == "SHIFT"

    print(f"Discarded {num_non_proj} non-projective sentences.")
    return configs, ops

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


def classify_with_pre_conds(input, stack, dependencies, svm_classifier, vectoriser):
    # Set up the parser configuration dictionary
    config = {
        "input": input,
        "stack": stack
    }

    # Extract the features
    sparse_feats, dense_feats = extract_vector_features(config)

    # Vectorize sparse features using the trained vectoriser
    sparse_features = vectoriser.transform([sparse_feats])

    # Convert dense features to a sparse matrix
    dense_features = scipy.sparse.csr_matrix(dense_feats)

    # Combine features into a sparse matrix
    combined_features = scipy.sparse.hstack([sparse_features, dense_features])

    # Compute the probabilities of all three classes
    probs = svm_classifier.predict_proba(combined_features)

    # Store the probabilities for each class
    classifier_result = {c: p for c, p in zip(svm_classifier.classes_, probs[0])}
    print(f"Initial probabilities: {classifier_result}")

    # Disallow the LEFT-ARC operation if the top element is ROOT
    if len(stack) > 0 and stack[-1][0] == 0:
        classifier_result["LEFT-ARC"] = 0

    # Disallow LEFT-ARC or RIGHT-ARC based on conditions
    for dep, head in dependencies:
        # Disallow LEFT-ARC if top element in the stack is already a dependent
        if len(stack) > 0 and stack[-1][0] == dep:
            classifier_result["LEFT-ARC"] = 0

        # Disallow RIGHT-ARC if the first element in the input already has a head
        if len(input) > 0 and input[0][0] == dep:
            classifier_result["RIGHT-ARC"] = 0

    # Return the most likely operation
    return max(classifier_result, key=classifier_result.get)

def dependency_parse(sentence, svm_classifier, vectoriser):
    # Initialise the stack with ROOT node
    stack = [(0, "ROOT", "_", "ROOT_POS", np.zeros(config.NO_OF_DIMENSIONS))]

    # Transform sentence into input (index, word, pos_tag) and add vector embeddings
    input = [(i + 1, word, lemma, pos_tag, get_vector(word)) for i, (word, lemma, pos_tag, _) in enumerate(sentence)]

    # Initialise the list of dependencies
    dependencies = []

    while len(input) > 0:
        # Classify with preconditions to get the operation
        op = classify_with_pre_conds(input, stack, dependencies, svm_classifier, vectoriser)
        print(f"Operation: {op}")

        # Perform operations (SHIFT, LEFT-ARC, RIGHT-ARC)
        if op == "SHIFT":
            # Shift element from input to stack
            stack.append(input.pop(0))
        elif op == "LEFT-ARC":
            # Make top element on stack a dependent of the first element in the input
            dep_idx, dep_word, dep_lemma, dep_embedding, dep_pos = stack.pop()
            head_idx, head_word, head_lemma, head_embedding, head_pos = input[0]
            dependencies.append((dep_idx, head_idx))
        elif op == "RIGHT-ARC":
            # Make the first item in the input a dependent of the top element on the stack
            head_idx, head_word, head_lemma, head_embedding, head_pos = stack.pop()
            dep_idx, dep_word, dep_lemma, dep_embedding, dep_pos = input[0]
            dependencies.append((dep_idx, head_idx))
            input[0] = (head_idx, head_word, head_lemma, head_embedding, head_pos)

    # Sort the dependencies by the index of the dependent
    sorted_dependencies = sorted(dependencies, key=lambda edge: edge[0])
    return sorted_dependencies

config.TRAIN_CONFIGS, config.TRAIN_OPS = simulate_parsing(config.TRAINING_SENTENCES)
