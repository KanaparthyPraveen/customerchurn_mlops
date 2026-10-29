from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

from src.preprocessing import load_and_preprocess_data


def run_tracking():
    print("=" * 50)
    print("STARTING MLFLOW EXPERIMENT TRACKING")
    print("=" * 50)

    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    db_path = project_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Customer Churn Prediction")

    with mlflow.start_run(run_name="GradientBoosting_Pipeline"):
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

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="churn_model",
            serialization_format="cloudpickle"
        )

        print("Logged Model: GradientBoostingClassifier")
        print(f"Logged Test Accuracy: {accuracy:.4f}")
        print("MLflow tracking completed successfully!")


if __name__ == "__main__":
    run_tracking()
