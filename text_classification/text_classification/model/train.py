from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# def train_model(data_path: str, model_path: str):
#     # Load dataset
#     data = pd.read_csv(data_path)
#     X, y = data['text'], data['label']

#     # Split into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Create pipeline
#     pipeline = Pipeline([
#         ('tfidf', TfidfVectorizer()),
#         ('svm', SVC(kernel='linear', probability=True))
#     ])

#     # Train model
#     pipeline.fit(X_train, y_train)

#     # Test model
#     y_pred = pipeline.predict(X_test)
#     print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

#     # Save model
#     joblib.dump(pipeline, model_path)
#     print(f"Model saved at {model_path}")
# if __name__ == "__main__":
#     train_model("sample_data.csv", "svm_model.pkl")

import os

def train_model(data_path: str, model_path: str):
    # Load dataset
    data = pd.read_csv(data_path)
    X, y = data['Text'], data['Label']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', SVC(kernel='linear', probability=True))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Test model
    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save model
    joblib.dump(pipeline, model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    # Ensure paths are relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "sample_data.csv")
    model_path = os.path.join(script_dir, "svm_model.pkl")
    
    train_model(data_path, model_path)
