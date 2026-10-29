from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

from src.preprocessing import load_and_preprocess_data


def train_model():
    print("=" * 50)
    print("STARTING MODEL TRAINING")
    print("=" * 50)

    X, y = load_and_preprocess_data()
    print(f"Dataset shape: {X.shape}, Target distribution: {dict(y.value_counts())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ))
    ])

    print("Fitting Gradient Boosting Pipeline...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print("\n" + "=" * 50)
    print("MODEL TRAINING RESULTS")
    print("=" * 50)
    print(f"Test Accuracy:        {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"Cross-Val Accuracy:   {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"CV Fold Scores:       {[round(float(s), 4) for s in cv_scores]}")
    print()
    print(classification_report(y_test, predictions, target_names=["No Churn", "Churn"]))

    model_path = project_path / "models" / "churn_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}!")

    return accuracy


if __name__ == "__main__":
    train_model()
