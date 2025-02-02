import numpy as np
import config

def read_conllu_file(file_path):
    """
    Reads a CoNLL-U file and returns a list of sentences, 
    where each sentence is represented as a list of tuples containing 
    word, lemma, POS tag, and dependency head.

    Args:
        file_path (str): Path to the CoNLL-U file.

    Returns:
        list: A list of sentences, where each sentence is a list of tuples 
              (word, lemma, POS tag, dependency head).
    """
    sentences = []

    with open(file_path, "r", encoding="UTF-8") as in_f:
        current_sentence = []

        for line in in_f:
            line = line.strip()

            # Ignore lines starting with '#' (comments)
            if line.startswith("#"):
                continue

            # An empty line indicates the end of the current sentence
            if line == "":
                if current_sentence:  # Avoid appending empty sentences
                    sentences.append(current_sentence)
                current_sentence = []
                continue

            # Split the line into its parts
            parts = line.split("\t")

            # Extract the index (the first column)
            idx = parts[0]

            # Skip multi-word tokens and empty nodes
            if "." in idx or "-" in idx:
                continue

            # Ensure there are enough columns in the line
            if len(parts) < 4:
                print(f"Skipping line with insufficient columns: {line}")  # Debugging statement

            # Extract the word, lemma, tag, and dependency head
            try:
                word = parts[1]
                lemma = parts[2]
                pos_tag = parts[3]
                dep_head = int(parts[6])  # Convert dependency head to an integer
                current_sentence.append((word, lemma, pos_tag, dep_head))
            except IndexError:
                print(f"Error processing line: {line}")  # Handle unexpected format

    return sentences

def load_glove_embeddings():
    """
    Loads GloVe embeddings from a file and returns the embedding matrix.
    
    Args:
        file_path (str): Path to the GloVe file.

    Returns:
        np.ndarray: The matrix of word embeddings.
    """
    embeddings = np.loadtxt(config.embeddings_file_path, encoding="UTF-8", usecols=config.COLS_TO_USE, comments=None)
    print(f"Shape of embeddings matrix: {embeddings.shape}")
    return embeddings

def create_word_to_index_mapping():
    """
    Creates a mapping from words to their corresponding index in the embeddings file.
    
    Args:
        file_path (str): Path to the GloVe file.

    Returns:
        dict: A dictionary mapping each word to its index.
    """
    word2index = {}
    with open(config.GLOVE_FILE_PATH, "r", encoding="UTF-8") as embedding_f:
        for i, line in enumerate(embedding_f):
            cols = line.split(" ")  # Split the line into columns
            word = cols[0]  # The first column contains the word
            word2index[word] = i  # Map the word to its index
    print(f"word2index has {len(word2index)} entries.")
    return word2index

def get_unknown_word_vector(embeddings):
    """
    Computes the vector for unknown words by taking the mean of all word vectors in the embedding matrix.
    
    Args:
        embeddings (np.ndarray): The matrix of word embeddings.
    
    Returns:
        np.ndarray: The vector representing unknown words.
    """
    unk_vector = np.mean(embeddings, axis=0)
    print(f"Computed word vector of shape {unk_vector.shape}")
    return unk_vector

def get_vector(word):
    """
    Returns the word vector from the embedding matrix.

    If the word is not found in the embedding matrix, returns the special UNK_VECTOR.

    Args:
        word (str): The word to look up in the embedding matrix.

    Returns:
        numpy.ndarray: The corresponding word vector or UNK_VECTOR if the word is not found.
    """
    word = word.lower()
    if word not in config.WORD2INDEX:
        return config.UNK_VECTOR
    else:
        # Look up the correct row in the embeddings matrix
        idx = config.WORD2INDEX[word]
        return embeddings[idx, :]

# Define UNK_VECTOR after loading embeddings
embeddings = load_glove_embeddings()
config.UNK_VECTOR = get_unknown_word_vector(embeddings)

# Create the word-to-index mapping and store it in config
config.WORD2INDEX = create_word_to_index_mapping()

# Load training sentences
config.TRAINING_SENTENCES = read_conllu_file(config.CONLLU_FILE_PATH)