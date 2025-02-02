from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import config

# Assuming 'config.COMBINED_FEATURES' contains all features and 'config.TRAIN_OPS' contains the target labels

# Split the data into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(config.COMBINED_FEATURES, config.TRAIN_OPS, test_size=0.2, random_state=42)

# Initialize a new SVM classifier with probability estimates
svm_classifier = SVC(kernel='linear', probability=True, max_iter=4000)

# Train the classifier on the training data
svm_classifier.fit(X_train, y_train)

# Get predictions on the test set
test_preds = svm_classifier.predict(X_test)

# Compute accuracy using accuracy_score
test_accuracy = accuracy_score(y_test, test_preds) * 100

# Print the test accuracy
print(f"Test Accuracy: {test_accuracy:.2f}%")

