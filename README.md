# House-price-prediction_Kubeflow_Azure_devops_CI-CD
Docker, Kubernetes, FastAPI, MLflow, Kubeflow

house-price-mlops/
├── .gitignore
├── README.md
├── requirements.txt
├── params.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── reference/
│   └── current/
├── artifacts/
│   ├── models/
│   └── reports/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── feature_engineering.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── pipeline.py
│   ├── drift_detection.py
│   ├── retraining_trigger.py
│   └── utils.py
├── api/
│   ├── app.py
│   ├── model_loader.py
│   ├── schemas.py
│   ├── requirements.txt
│   └── Dockerfile
├── mlflow/
│   └── Dockerfile
├── docker-compose.yml
├── k8s/
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   ├── mlflow-deployment.yaml
│   └── mlflow-service.yaml
└── helm/
    └── house-price-api/

# Local env setup:

python -m venv .venv
.venv/Scripts/activate.ps1
pip install -r requirements.txt
python.exe -m pip install --upgrade pip

Python -m pip install --upgrade pip setuptools wheel
sleep 5    # This is to add Sleep of 5 seconds to ensure that pip, setuptools and wheel are upgraded before we run pip install pip-tools
pip install pip-tools 
sleep 5 # # This is to add Sleep of 5 seconds to ensure that pip-tools is installed before we run pip-compile
pip-compile requirements.in  #this generates requirements.txt, then install requirements.txt
pip install -r requirements.txt

to upgrade:

pip-compile --upgrade-package <package-name> requirements.in
pip-compile --upgrade-package scikit-learn requirements.in
pip-compile --upgrade requirements.in


# Now all py notebooks are ready inside src folder

Module	Role
data_ingestion.py	step 1
data_validation.py	step 2
preprocess.py	step 3
train.py	step 4
evaluate.py	step 5
pipeline.py	orchestrates all


First Connect to mlflow ui:

mlflow ui --host localhost --port 5000
python -m src.pipeline

Interpretation of metrics:

MAE: Less is Good
MSE: Less is Good
RMSE: Less is Good
r2 score: as ggod as near to 1 (how well model able to explain the variation)

# Great expectation validation:

add great_expectations into requirements.in

echo "great_expectations" >> requirements.in
pip-compile requirements.in
pip install -r requirements.txt

Initilize project

python -m great_expectations

# data Versioning

dvc init
