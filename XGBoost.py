"""
XGBoost Classification Example using the Iris Dataset

This script demonstrates:
1. Loading and preprocessing the Iris dataset.
2. Training an XGBoost classifier.
3. Evaluating the model.
4. Saving the trained model.
5. Loading and using the saved model for predictions.

Dependencies:
- xgboost
- scikit-learn
- pandas
- numpy
- joblib

To install the dependencies, use:
pip install xgboost scikit-learn pandas numpy joblib
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib

# Step 1: Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(url, header=None, names=column_names)

# Display the first few rows of the dataset to understand its structure
print("First few rows of the Iris dataset:")
print(df.head())

# Display basic statistics about the dataset
print("\nBasic statistics of the dataset:")
print(df.describe())

# Display the distribution of the target class
print("\nClass distribution:")
print(df['class'].value_counts())

# Step 2: Preprocess the data
# Encode the target labels (classes) to numeric values
df['class'] = df['class'].astype('category').cat.codes

# Define features (X) and target (y)
X = df.drop('class', axis=1)
y = df['class']

# Split the data into training and testing sets
# 70% for training, 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Display the size of training and testing sets
print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Step 3: Create and train an XGBoost classifier
# Define the model with basic parameters
model = XGBClassifier(
    objective='multi:softprob',  # For multi-class classification
    num_class=3,  # Number of classes in the Iris dataset
    eval_metric='mlogloss',  # Metric used for evaluation
    use_label_encoder=False,  # Disable label encoder as it is deprecated
    random_state=42  # For reproducibility
)

# Train the model on the training data
print("\nTraining the XGBoost classifier...")
model.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate the classification report
class_report = classification_report(y_test, y_pred, target_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

print("\nModel evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Step 6: Save the trained model to a file
model_file = "xgboost_iris_model.json"
joblib.dump(model, model_file)
print(f"\nModel saved to {model_file}")

# Demonstrate loading the model and making a prediction
# Load the model from the file
loaded_model = joblib.load(model_file)
print("\nLoaded model from file.")

# Predict on a sample data point
sample_data = X_test.iloc[0:1]
predicted_class = loaded_model.predict(sample_data)[0]
actual_class = y_test.iloc[0]

print("\nPrediction on a sample data point:")
print(f"Sample features: {sample_data.values}")
print(f"Predicted class: {predicted_class}, Actual class: {actual_class}")

# Detailed breakdown of what the classifier sees
print("\nFeature importances from the model:")
print(model.get_booster().get_score(importance_type='weight'))
