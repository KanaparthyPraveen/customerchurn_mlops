from pathlib import Path
import sys
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

from src.preprocessing import load_and_preprocess_data


def evaluate():
    print("=" * 50)
    print("STARTING MODEL EVALUATION")
    print("=" * 50)

    X, y = load_and_preprocess_data()
    model_path = project_path / "models" / "churn_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run src/train.py first.")

    model = joblib.load(model_path)
    predictions = model.predict(X)

    acc = accuracy_score(y, predictions)
    print(f"Overall Dataset Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))
    print("\nClassification Report:")
    print(classification_report(y, predictions, target_names=["No Churn", "Churn"]))


if __name__ == "__main__":
    evaluate()
