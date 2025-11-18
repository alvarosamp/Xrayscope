import os
import subprocess
import pickle
from datetime import datetime
import logging
import boto3
import watermark
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from utils.bucket_utils import wait_for_bucket
from utils.config_reader import load_config
from utils.data_utils import load_data
from dotenv import load_dotenv

#Load  .env and AWs credentials
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

#configurando logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model-init')

def train_model(X, y, test_size: float = 0.2, config_path: str = 'config.yaml'):
    """
    Treina um modelo RandomForestClassifier com os dados fornecidos.

    Args:
        X: Features (np.array).
        y: Labels (np.array).
        test_size: Proporção dos dados para teste (default é 0.2).
        config_path: Caminho para o arquivo de configuração YAML.

    Returns:
        modelo treinado, acurácia no conjunto de teste
    """
    logger.debug(f'Carregando a configuraocao do arquivo: {config_path}')
    config = load_config(config_path)

    if "model" not in config:
        raise ValueError("Configuração inválida: seção 'model' ausente.")
    model_conf = config["model"]

    model_name = model_conf.get("name")
    hyperparameters = model_conf.get("hyperparameters")
    if not model_name or not hyperparameters:
        raise ValueError("Configuração inválida: 'name' ou 'hyperparameters' ausentes na seção 'model'.")
    random_state = hyperparameters.get("random_state")
    if random_state is None:
        raise ValueError("The 'random_state' parameter must be defined under 'model.hyperparameters'.")
    logger.info(f'Treinando o modelo: {model_name} com hiperparâmetros: {hyperparameters}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


    logger.info(
        f"Instantiating model {model_name} with hyperparameters {hyperparameters}"
    )
    if model_name == "RandomForest":
        model = RandomForestClassifier(**hyperparameters)
    else:
        raise ValueError(f"Unsupported model '{model_name}' in config.")

    logger.info(f"Fitting model on {X_train.shape[0]} samples")
    time_start = datetime.now()
    model.fit(X_train, y_train)
    time_end = datetime.now()
    logger.info(f"Model training completed in {time_end - time_start}")
    logger.info(f"Predicting on test set with {X_test.shape[0]} samples")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["NORMAL", "PNEUMONIA"], output_dict=True
    )

    version_info = watermark.watermark(
        packages="numpy,scipy,pandas,scikit-learn,opencv-python,boto3"
    )
    metricas = {
        'accuracy': accuracy,
        'classification_report': report,
        'version_info': version_info,
        'split':{'train': X_train.shape[0],'test': X_test.shape[0]},
        'time_training': (time_end - time_start).total_seconds()
    }
    logger.info(f"Trainamento completo. Metricas : {metricas}")
    return model, metricas

def save_model_to_bucket(s3,bucket_name : str, model, file_name:str):
    try:
        data = pickle.dumps(model)
        s3.put_object(Bucket = bucket_name, Key = file_name, Body = data)
        logger.info(f'Modelo salvo no bucket {bucket_name} com a chave {file_name}')
    except Exception as e:
        logger.error(f"Erro ao salvar o modelo no bucket: {e}")
        raise

def main():
    execution_env = os.getenv("EXECUTION_ENV", "local")
    logger.info(f'Execution environment: {execution_env}')
    try:
        #Configurando s3 com base no ambiente
        if execution_env == 'cloud':
            s3 = boto3.client(
                's3',
                aws_access_key_id = AWS_ACCESS_KEY_ID,
                aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
            )
        else:
            s3 = boto3.client(
                's3',
                aws_access_key_id = AWS_ACCESS_KEY_ID,
                aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
                endpoint_url=MLFLOW_S3_ENDPOINT_URL
            )
    
        datasource_bucket = os.getenv('Datasource_bucket', 'datasource')
        dev_models_bucket = os.getenv('Dev_models_bucket', 'dev-models')
        img_size = (64, 64)

        for bucket in (datasource_bucket, dev_models_bucket):
            if not wait_for_bucket(s3, bucket):
                msg = f'Bucket {bucket} not available after waiting period.'
                logger.error(msg)
                raise ValueError(msg)
        
        X, y = load_data(s3, datasource_bucket, img_size)
        logger.info(f'Dados carregados: {len(X)} amostras.')
        model, metrics = train_model(X, y, test_size = 0.2)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'model_{timestamp}.pkl'
        save_model_to_bucket(s3, dev_models_bucket, model, file_name)
        auto_register = os.getenv("AUTO_REGISTER", "false").lower() == "true"
        logger.info(f"AUTO_REGISTER flag is {auto_register}")
        if auto_register:
            logger.info("Calling model_reg.py")
            result = subprocess.run(
                ["python", "model_reg.py"], capture_output=True, text=True
            )
            logger.info(
                f"model_reg.py output -- stdout: {result.stdout}, stderr: {result.stderr}, returncode: {result.returncode}"
            )
            if result.returncode != 0:
                logger.error(f"model_reg.py failed with return code {result.returncode}")
        else:
            logger.info("Skipping registration (AUTO_REGISTER disabled)")

    except Exception:
        logger.exception("Unhandled exception in main()")
        raise
    finally:
        logger.info("Script end")


if __name__ == "__main__":
    main()
