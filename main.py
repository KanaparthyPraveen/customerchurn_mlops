"""
Customer Churn Prediction - Standard Application Entrypoint
Run this script to launch the unified FastAPI web service and frontend interface.
"""
import sys
import threading
import webbrowser
from pathlib import Path
import uvicorn

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.main import app


def open_browser():
    webbrowser.open("http://127.0.0.1:8000/")


def main():
    print("=" * 65)
    print(" CUSTOMER CHURN PREDICTION SYSTEM")
    print("=" * 65)
    print(" Starting unified application (Backend + Frontend)...")
    print(" Web UI (Home)     : http://127.0.0.1:8000/")
    print(" Prediction Form   : http://127.0.0.1:8000/prediction")
    print(" API Documentation : http://127.0.0.1:8000/docs")
    print(" Health Check      : http://127.0.0.1:8000/health")
    print("=" * 65)

    threading.Timer(1.5, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
