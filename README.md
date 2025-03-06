# Dependency Parser 

This project implements a **dependency parser** that extracts vector features from a given input sentence and stack configuration. It uses word embeddings (such as GloVe) and linguistic features to generate feature vectors for a machine learning model. The model is trained using **Support Vector Machines (SVM)** for parsing tasks.

## Project Overview 

The parser is designed to work with sentence data in **CONLL-U format** and extracts features based on the stack and input words of a shift-reduce parsing algorithm. It leverages word embeddings and linguistic features (like POS tags and lemmatisation) for effective training of a machine learning model. The parser uses a linear kernel **SVM** model, and the trained model can be used to predict syntactic dependencies between words in a sentence.

## Files and Structure 

- `config.py`: Configuration file storing constants such as dimensions for word embeddings, training data paths, and parser configurations.
- `utils.py`: Utility functions for reading data, loading embeddings, and processing vectors.
- `parser.py`: The core script that extracts features and trains the model.
- `training.py`: This script trains the SVM classifier on the combined feature set.

## Features 

- Extracts word embeddings for input and stack words.
- Handles multiple linguistic features like form, lemma, and POS tags.
- Uses a **linear kernel SVM** for classification tasks.
- Supports handling of sparse and dense feature matrices.

## Requirements 

- `scikit-learn`
- `numpy`
- `scipy`

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## How to Run 

If you'd like to try out my projects, follow these steps:

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/kanincityy/dependency_parser.git
    ```
2. Navigate to the specific project's directory.
3. Run the code or open the Jupyter Notebooks to explore the results.

## Contributing 

This project is a reflection of my learning, but feel free to fork the repository and contribute if you have ideas or improvements!

## License 

This repository is licensed under the MIT License. See the LICENSE file for details.

---

Happy coding! ✨🐇
