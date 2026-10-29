from pathlib import Path
import pandas as pd

project_path = Path(__file__).resolve().parent.parent


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature transformations and engineering consistent with model training."""
    df = df.copy()

    # Map categoricals (handles both string and numeric inputs)
    if "gender" in df.columns:
        gender_map = {"Male": 1, "Female": 0, "male": 1, "female": 0, 1: 1, 0: 0, "1": 1, "0": 0}
        df["gender"] = df["gender"].map(lambda x: gender_map.get(x, x)).astype(int)

    if "contract" in df.columns:
        contract_map = {
            "Month-to-month": 0, "One year": 1, "Two year": 2,
            "month-to-month": 0, "one year": 1, "two year": 2,
            0: 0, 1: 1, 2: 2, "0": 0, "1": 1, "2": 2
        }
        df["contract"] = df["contract"].map(lambda x: contract_map.get(x, x)).astype(int)

    if "internet_service" in df.columns:
        internet_map = {
            "DSL": 0, "Fiber optic": 1,
            "dsl": 0, "fiber optic": 1,
            0: 0, 1: 1, "0": 0, "1": 1
        }
        df["internet_service"] = df["internet_service"].map(lambda x: internet_map.get(x, x)).astype(int)

    if "senior_citizen" in df.columns:
        df["senior_citizen"] = df["senior_citizen"].astype(int)

    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].astype(int)

    if "monthly_charges" in df.columns:
        df["monthly_charges"] = df["monthly_charges"].astype(float)

    # Engineered features
    df["short_tenure_mtm"] = (
        (df["tenure"] <= 12) & (df["contract"] == 0)
    ).astype(int)

    df["fiber_high_charge"] = (
        (df["internet_service"] == 1) & (df["monthly_charges"] >= 70)
    ).astype(int)

    def get_bucket(t):
        if t <= 6:
            return 0
        elif t <= 12:
            return 1
        elif t <= 24:
            return 2
        elif t <= 48:
            return 3
        return 4

    df["tenure_bucket"] = df["tenure"].apply(get_bucket)
    df["charge_per_tenure"] = df["monthly_charges"] / (df["tenure"] + 1)

    return df


def load_and_preprocess_data(csv_path: Path = None):
    """Load dataset from CSV and return preprocessed feature matrix X and target y."""
    if csv_path is None:
        csv_path = project_path / "data" / "customer_churn.csv"

    data = pd.read_csv(csv_path)

    if "customer_id" in data.columns:
        data = data.drop("customer_id", axis=1)

    if "churn" in data.columns:
        churn_map = {"No": 0, "Yes": 1, "no": 0, "yes": 1, 0: 0, 1: 1, "0": 0, "1": 1}
        y = data["churn"].map(lambda x: churn_map.get(x, x)).astype(int)
        X_raw = data.drop("churn", axis=1)
        X = preprocess_features(X_raw)
        return X, y
    else:
        return preprocess_features(data), None


if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    print("=" * 50)
    print("Preprocessed Data shape:", X.shape)
    print("Features:", list(X.columns))
    print("\nFirst 3 rows:")
    print(X.head(3))
    print("\nTarget distribution:")
    print(y.value_counts())
