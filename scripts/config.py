# ===========================
# File Paths Configuration
# ===========================
# Path to the CoNLL-U file (training data)
CONLLU_FILE_PATH = "data/en_ewt-ud-train.conllu"

# Path to the GloVe embedding file
GLOVE_FILE_PATH = "glove/glove.6B.300d.txt"

# ===========================
# Embedding Configuration
# ===========================
# Number of dimensions in the GloVe embeddings
NO_OF_DIMENSIONS = 300

# Columns to use for loading embeddings (1 to 300 in this case)
COLS_TO_USE = list(range(1, NO_OF_DIMENSIONS + 1))

# ===========================
# Special Constants
# ===========================
# Unknown vector (will be defined in the script or elsewhere)
UNK_VECTOR = None  # To be defined later after loading embeddings
WORD2INDEX = None # To store the word-to-index mapping
TRAINING_SENTENCES = None # Placeholder for training data
TRAIN_CONFIGS, TRAIN_OPS = None, None # Placeholder for simulator parser configurations
COMBINED_FEATURES = None # Placeholder for combined feature matrix

