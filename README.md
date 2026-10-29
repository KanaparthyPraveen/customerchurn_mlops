# Customer Churn Prediction - MLOps

An end-to-end Machine Learning Operations (MLOps) project that predicts customer churn probability and risk levels based on customer account parameters, contracts, and usage patterns.

The project features a high-performance **Gradient Boosting** classification pipeline, an interactive **anti-slop web frontend**, and a **FastAPI** backend service with full test coverage and MLflow experiment tracking.

---

## 🚀 Quick Start (Standard Commands)

### 1. Clone the Repository
```bash
git clone https://github.com/KanaparthyPraveen/customerchurn_mlops.git
cd customerchurn_mlops
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run the Project

### Standard Command (Backend + Frontend Unified)
The frontend HTML interfaces are served directly through FastAPI. Running the standard entrypoint starts both the backend API and serves the interactive frontend web application in one command:

```bash
python main.py
```

*Your browser will automatically open to `http://127.0.0.1:8000/`.*

### Alternative Commands

* **Using Uvicorn with auto-reload (Development Mode):**
  ```bash
  uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
  ```

* **Running the FastAPI application module directly:**
  ```bash
  python app/main.py
  ```

---

## 🌐 Application URL Reference

Once the server is running, access the following endpoints:

| Interface | URL | Description |
|---|---|---|
| **Landing Page** | [http://127.0.0.1:8000/](http://127.0.0.1:8000/) | Clean, anti-slop overview with model metrics and schema |
| **Prediction Form** | [http://127.0.0.1:8000/prediction](http://127.0.0.1:8000/prediction) | Interactive form to input customer details and run inference |
| **Prediction Result** | [http://127.0.0.1:8000/result](http://127.0.0.1:8000/result) | Visual risk score, probability, risk breakdown, and actions |
| **API Documentation** | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | Interactive Swagger UI for testing API endpoints |
| **Health Check** | [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) | API health and status check endpoint |

---

## 🛠️ Machine Learning Pipeline Commands

All data processing, model training, evaluation, and experiment tracking scripts are organized in the `src/` directory:

### 1. Data Preprocessing & Feature Engineering
Inspect and test dataset encoding and engineered features:
```bash
python src/preprocessing.py
```

### 2. Train the Model
Train the Gradient Boosting pipeline on the 6,000-record dataset and save the serialized model to `models/churn_model.pkl`:
```bash
python src/train.py
```
*Current test accuracy: **90.0%** with 5-fold cross-validation.*

### 3. Model Evaluation
Evaluate the trained model on the full dataset, computing classification metrics and confusion matrix:
```bash
python src/evaluate.py
```

### 4. MLflow Experiment Tracking
Log model hyperparameters, accuracy metrics, and artifacts to MLflow:
```bash
python src/mlflow_tracking.py
```
To view the MLflow dashboard:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then visit [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 🧪 Running Unit Tests

Run the automated test suite with pytest:
```bash
pytest -v
```

---

## 📊 Dataset Schema (`data/customer_churn.csv`)

The dataset contains 6,000 customer records with balanced churn patterns:

| Feature | Type | Range / Values | Description |
|---|---|---|---|
| `customer_id` | String | `C0001` - `C6000` | Unique customer identifier |
| `gender` | String / Int | `Male` (1), `Female` (0) | Account holder gender |
| `senior_citizen` | Int | `0`, `1` | Senior citizen indicator (65+ years) |
| `tenure` | Integer | `1` - `72` | Months subscribed with the service |
| `monthly_charges`| Float | `18.0` - `120.0` | Recurring monthly bill in USD |
| `contract` | String / Int | `Month-to-month` (0), `One year` (1), `Two year` (2) | Account commitment type |
| `internet_service` | String / Int | `DSL` (0), `Fiber optic` (1) | Internet connection type |
| `churn` | String / Int | `Yes` (1), `No` (0) | Target variable: whether customer cancelled |

---

## 📁 Project Structure

```text
customerchurn_mlops/
├── app/
│   └── main.py              # FastAPI application & prediction endpoints
├── data/
│   └── customer_churn.csv   # 6,000-row customer dataset
├── frontend/
│   ├── index.html           # Anti-slop landing page
│   ├── prediction.html      # Customer information & prediction form
│   └── result.html          # Prediction outcome & retention recommendations
├── models/
│   └── churn_model.pkl      # Trained Gradient Boosting pipeline artifact
├── src/
│   ├── preprocessing.py     # Data transformation & feature engineering
│   ├── train.py             # Model training with cross-validation
│   ├── evaluate.py          # Comprehensive evaluation & confusion matrix
│   └── mlflow_tracking.py   # MLflow experiment tracking
├── tests/
│   └── test_api.py          # Automated API and endpoint unit tests
├── .gitignore               # Ignored files & directories
├── main.py                  # Standard unified entrypoint
├── requirements.txt         # Core dependencies
└── README.md                # Project documentation
```
