import os 
import re 
import pickle 
import logging 
import boto3
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from utils.config_reader import load_config
from utils.data_utils import load_data
from utils.bucket_utils import wait_for_bucket

#Load .env and AWS credentials
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

#Configure standard logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model-registry')

def load_model_from_bucket(s3, bucket_name: str, specific_filename: str = None):
    """
    Carrega o modelo mais recente do bucket S3/MinIO.

    Args:
        s3: cliente boto3 S3.
        bucket_name: nome do bucket.
        specific_filename: nome específico do arquivo de modelo (opcional).
    Returns:
        modelo carregado
    """
    logger.info(f"Carregando modelo do bucket: {bucket_name}")
    if specific_filename:
        response = s3.get_object(Bucket=bucket_name, Key=specific_filename)
        logger.info(f"Carregando modelo específico: {specific_filename}")
        return pickle.loads(response['Body'].read())
    response = s3.list_objects_v2(Bucket=bucket_name)

    if 'Contents' not in response:
        msg = f'No file found in the bucket {bucket_name}'
        logger.error(msg)
        raise ValueError(msg)

    pattern = re.compile(r'model_(\d{8}_\d{6})\.pkl')
    models = [
        (obj["Key"], pattern.match(obj["Key"]).group(1))
        for obj in response["Contents"]
        if pattern.match(obj["Key"])
    ]     
    if not models:
        msg = f'No model files found in the bucket {bucket_name}'
        logger.error(msg)
        raise ValueError(msg)

    #Pick the latest by timestamp
    models.sort(key = lambda x: x[1], reverse = True)
    selected = models[0][0]
    logger.info(f"Modelo selecionado: {selected}")
    data = s3.get_object(Bucket = bucket_name, Key = selected)["Body"].read()
    return pickle.loads(data)

def evaluate_model(model, X, y, test_size : float, random_state: int = 42):
    logger.info("Avaliando o modelo carregado")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy}")
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{report}")
    return accuracy, report

def should_register_as_experiment_only(auto: bool, accuracy: float) -> bool:
    """
    Decide se o modelo deve ser registrado apenas como experimento
    ou também na model registry, baseado na acurácia.

    Args:
        auto: se True, usa o limiar de acurácia para decidir.
        accuracy: acurácia do modelo.
    Returns:
        True se deve registrar apenas como experimento, False se deve registrar na model registry.
    """
    if auto and accuracy < 0.5:
        logger.warning(f'Accuracy {accuracy} below threshold, registering as experiment only.')
        return True
    if not auto:
        # interactive fallback
        choice = input("Promote to Production? (Yes/No): ").strip().lower()
        do_experiment = (choice != "yes")
        logger.info(f"User chose to promote: {not do_experiment}")
        return do_experiment

    return False

def register_model(model, accuracy: float, report: dict, promote_to_production: bool, execution_env: str, config_path: str = "config.yaml"):
    logger.info(f"Starting MLflow registration. Promote to production: {promote_to_production}")
    config = load_config(config_path)
    model_name = config.get("model", {}).get("name")
    if not model_name:
        msg = "Config missing model.name"
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Setting MLflow registry URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
    logger.info(f"Setting MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("Setting experiment: Model_Experiment")
    mlflow.set_experiment("Model_Experiment")
    logger.info("MLflow configuration completed")

    with mlflow.start_run() as run:
        hyperparameters = config.get("model", {}).get("hyperparameters", {})
        mlflow.log_param("n_estimators", hyperparameters.get("n_estimators"))
        mlflow.log_param("environment", execution_env)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_dict(report, "classification_report.json")
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        logger.info(f"Model logged to MLflow with run_id: {run.info.run_id}")

        if promote_to_production:
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                logger.info(
                    f"Promoted model version {latest.version} to Production"
                )
            else:
                logger.warning(f"No model versions found for model '{model_name}' to promote.")
        else:
            logger.info(f"Registered model '{model_name}' as experiment only.")


def main(specific_model_name: str = None, auto: bool = True):
    execution_env = os.getenv("EXECUTION_ENVIRONMENT", "local")
    logger.info(f"Script start. Auto mode: {auto}, Execution: {execution_env}")
    logger.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Environment variables: {dict(os.environ)}")
    try:
        # Configure S3 client based on environment
        if execution_env == "cloud":
            s3 = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
        else:
            s3 = boto3.client(
                "s3",
                endpoint_url=MLFLOW_S3_ENDPOINT_URL,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )

        # Get bucket names from environment or use defaults
        datasource_bucket = os.getenv("DATASOURCE_BUCKET", "datasource")
        dev_models_bucket = os.getenv("DEV_MODELS_BUCKET", "dev-models")

        # ensure buckets exist
        for bucket in (dev_models_bucket, datasource_bucket):
            if not wait_for_bucket(s3, bucket):
                msg = f"Bucket '{bucket}' not found"
                logger.error(msg)
                raise ValueError(msg)

        model = load_model_from_bucket(s3, dev_models_bucket, specific_model_name)
        logger.info("Model loaded successfully")
        
        X, y = load_data(s3, datasource_bucket, (64, 64))
        logger.info("Data loaded successfully")
        
        accuracy, report = evaluate_model(model, X, y)
        logger.info(f"Model evaluated. Accuracy: {accuracy}")

        promote_to_prod = not should_register_as_experiment_only(auto, accuracy)
        logger.info(f"Will promote to production: {promote_to_prod}")
        
        try:
            register_model(model, accuracy, report, promote_to_prod, execution_env)
            logger.info("Model registration completed successfully")
        except Exception as e:
            logger.error(f"Model registration failed: {str(e)}")
            logger.exception("Full traceback:")
            raise

    except Exception:
        logger.exception("Unhandled exception in main()")
        raise
    finally:
        logger.info("Script end")


if __name__ == "__main__":
    import sys

    specific = None
    auto_flag = True
    if len(sys.argv) >= 2:
        specific = sys.argv[1]
    if len(sys.argv) >= 3:
        auto_flag = sys.argv[2].lower() == "true"

    main(specific_model_name=specific, auto=auto_flag)

