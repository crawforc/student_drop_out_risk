# ml/model.py - functions to train, test, save, and load the model

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_model(X_train, y_train):
    """
    Trains a Random Forest model and returns it.

    X_train - the features for the training students
    y_train - the answers (dropout yes/no) for the training students
    """
    model = RandomForestClassifier(
        n_estimators=100,   # build 100 decision trees and let them vote
        max_depth=10,       # limit how deep each tree grows (keeps the file small)
        random_state=42,    # same result every time we run it
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Compares the model's predictions to the real answers.

    y     - the real answers
    preds - the model's predictions

    Returns: accuracy, precision, recall, f1
    """
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    f1 = f1_score(y, preds, zero_division=1)
    return accuracy, precision, recall, f1


def inference(model, X):
    """
    Uses the model to make predictions (1 = likely to drop out, 0 = likely to stay).
    """
    preds = model.predict(X)
    return preds


def save_model(item, path):
    """
    Saves the model (or encoder) to a file so we can use it later.
    """
    joblib.dump(item, path)


def load_model(path):
    """
    Loads a saved model (or encoder) from a file.
    """
    return joblib.load(path)