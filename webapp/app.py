from flask import Flask, request, jsonify, render_template
import mlflow
try:
    from mlflow import sklearn as mlflow_sklearn
except Exception:
    # mlflow.sklearn may not be importable in some environments; fall back to None
    mlflow_sklearn = None
import numpy as np
import cv2
import time
import traceback
import logging
import sys
from mlflow.tracking import MlflowClient
from utils.logging_formatter import configure_fluent_logging

app = Flask(__name__)

# ---- Unified Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Configure Fluentd logging using our utility function
logger = configure_fluent_logging(
    logger_name="webapp",
    service_name="webapp",
    fluent_host="fluentd",
    fluent_port=24224
)


# Configure MLflow URIs from environment
import os
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Set headers for ALB routing if using ALB DNS
if "elb.amazonaws.com" in mlflow_uri:
    os.environ["MLFLOW_TRACKING_HEADERS"] = "Host: mlflow.hm-mlflow.local"

mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)

def get_current_model_info(model_name="RandomForest", stage="Production"):
    """
    Query MLflow to get the latest version of the model in a given stage.
    """
    def _get(v, field):
        # Mlflow may return model version objects or dicts depending on client version
        if isinstance(v, dict):
            return v.get(field)
        return getattr(v, field, None)

    try:
        client = MlflowClient()
        versions = client.search_model_versions("name='{}'".format(model_name))
        if versions:
            def _version_int(v):
                try:
                    return int(_get(v, "version") or -1)
                except Exception:
                    return -1

            sorted_versions = sorted(versions, key=_version_int, reverse=True)
            return {"name": model_name, "version": _get(sorted_versions[0], "version")}
    except Exception as e:
        logger.error("Error querying model", extra={
            "event": "model_query_error",
            "error": str(e)
        })
    return {"name": model_name, "version": "N/A"}

def wait_for_model_availability(model_name="RandomForest", stage="Production", timeout=600, poll_interval=10):
    """
    Wait until the model is available in MLflow in a given stage.
    Uses mlflow.sklearn.load_model to ensure access to predict_proba.
    """
    client = MlflowClient()
    elapsed = 0
    def _get(v, field):
        if isinstance(v, dict):
            return v.get(field)
        return getattr(v, field, None)

    while elapsed < timeout:
        try:
            logger.info("Checking model availability", extra={
                "event": "model_availability_check",
                "model": model_name,
                "stage": stage
            })
            versions = client.search_model_versions("name='{}'".format(model_name))
            if versions:
                def _version_int(v):
                    try:
                        return int(_get(v, "version") or -1)
                    except Exception:
                        return -1

                sorted_versions = sorted(versions, key=_version_int, reverse=True)
                latest_version = sorted_versions[0]
                logger.info("Found model versions: %s", [(_version_int(v), _get(v, "current_stage")) for v in sorted_versions])

                # Try Production stage first, then latest version
                production_versions = [v for v in versions if _get(v, "current_stage") == "Production"]
                if production_versions:
                    model_uri = f"models:/{model_name}/Production"
                    logger.info(f"Using Production stage: {model_uri}")
                else:
                    model_uri = f"models:/{model_name}/{_get(latest_version, 'version')}"
                    logger.info(f"Using latest version: {model_uri}")

                logger.info(f"Attempting to load model from URI: {model_uri}")
                # Prefer mlflow.sklearn loader when available (returns native sklearn estimator).
                if mlflow_sklearn is not None:
                    loaded_model = mlflow_sklearn.load_model(model_uri)
                else:
                    # Fallback to pyfunc which returns a pyfunc wrapper; this may not expose
                    # predict_proba depending on the model flavor, so we try to handle that later.
                    loaded_model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Successfully loaded model from URI: {model_uri}")
                logger.info("Model found", extra={
                    "event": "model_found",
                    "model": model_name,
                    "version": _get(sorted_versions[0], "version")
                })
                return loaded_model
        except Exception as e:
            logger.warning("Error checking model", extra={
                "event": "model_availability_warning",
                "error": str(e),
                "error_type": type(e).__name__
            })
            logger.error("Traceback: %s", traceback.format_exc())
        logger.info("Waiting before next check", extra={
            "event": "model_availability_wait",
            "wait_seconds": poll_interval
        })
        time.sleep(poll_interval)
        elapsed += poll_interval
    logger.critical("Timeout waiting for model", extra={
        "event": "model_timeout",
        "model": model_name,
        "stage": stage
    })
    return None

# Load model on startup - but don't block Flask startup
model = None

def load_model_async():
    """Load model in background thread"""
    global model
    try:
        logger.info("Starting background model loading")
        model = wait_for_model_availability(timeout=600, poll_interval=5)
        if model:
            logger.info("Model loaded successfully in background")
        else:
            logger.warning("Model not available after background loading")
    except Exception as e:
        logger.error(f"Error loading model in background: {e}")

# Start model loading in background thread
import threading
model_thread = threading.Thread(target=load_model_async, daemon=True)
model_thread.start()
logger.info("Webapp started - model loading in background")

@app.route("/model-info")
def get_model_info():
    # Query MLflow on each request to reflect the current version.
    info = get_current_model_info()
    logger.info("Informacoes do modelo recuperadas", extra={
        "event": "model_info",
        "info": info
    })
    # Retornar em português
    version = info.get("version") if info else "N/A"
    return jsonify({"modelo": info.get("name"), "versao": version})

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        logger.error("Tentativa de previsao sem modelo carregado", extra={
            "event": "prediction_failure",
            "reason": "no_model"
        })
        return jsonify({"erro": "Modelo não encontrado"}), 500
    try:
        data = request.get_json(silent=True)
        if not data or "features" not in data:
            raise ValueError("O JSON deve incluir o array 'features'.")

        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        # Mapear para rótulo legível
        label_map = {0: "Normal", 1: "Pneumonia"}
        label = label_map.get(int(prediction), str(prediction))
        logger.info("Previsao realizada com sucesso", extra={
            "event": "prediction",
            "result": int(prediction),
            "label": label
        })
        return jsonify({"previsao_codigo": int(prediction), "previsao_label": label})
    except Exception as e:
        logger.error("Erro durante previsao", extra={
            "event": "prediction_error",
            "error": str(e)
        })
        logger.error(traceback.format_exc())
        return jsonify({"erro": str(e)}), 400

@app.route("/diagnose", methods=["POST"])
def diagnose():
    global model
    if model is None:
        logger.error("Tentativa de diagnostico sem modelo carregado", extra={
            "event": "diagnosis_failure",
            "reason": "no_model"
        })
        return jsonify({"erro": "Modelo não encontrado"}), 500

    start_time = time.time()  # Start timing the diagnosis
    try:
        if 'image' not in request.files:
            logger.error("Nenhuma imagem fornecida para diagnostico", extra={
                "event": "diagnosis_error",
                "reason": "no_image"
            })
            return jsonify({"erro": "Nenhuma imagem enviada."}), 400
        
        file = request.files['image']
        image_bytes = file.read()
        np_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Não foi possível decodificar a imagem.")

        # Processar imagem para diagnostico
        img_resized = cv2.resize(img, (64, 64))
        features = img_resized.flatten().reshape(1, -1)
        pneumonia_prob = None
        diagnosis_str = ""

        # Prefer predict_proba when available; otherwise fall back to predict
        predict_proba_fn = getattr(model, "predict_proba", None)
        if callable(predict_proba_fn):
            probs = predict_proba_fn(features)
            probabilities = np.asarray(probs)[0]
            if len(probabilities) > 1:
                pneumonia_prob = float(probabilities[1])
                diagnosis_str = f"Probabilidade de Pneumonia: {pneumonia_prob*100:.2f}%"
            else:
                diagnosis_str = f"Probabilidades previstas: {probabilities.tolist()}"
        else:
            # Models loaded via mlflow.pyfunc may not expose predict_proba; use predict as fallback
            pred = model.predict(features)[0]
            label_map = {0: "Normal", 1: "Pneumonia"}
            label = label_map.get(int(pred), str(pred))
            diagnosis_str = f"Classe prevista: {label} ({int(pred)})"
        
        inference_time = time.time() - start_time  # Calculate inference time
        
        # Log diagnosis details with structured keys.
        logger.info("Diagnostico realizado com sucesso", extra={
            "event": "diagnosis",
            "diagnosis": diagnosis_str,
            "inference_time_ms": float(f"{inference_time*1000:.2f}")
        })
        return jsonify({"diagnostico": diagnosis_str, "tempo_inferencia_seg": inference_time})
    except Exception as e:
        logger.error("Erro durante diagnostico", extra={
            "event": "diagnosis_error",
            "error": str(e)
        })
        logger.error(traceback.format_exc())
        return jsonify({"erro": str(e)}), 400

@app.route("/feedback", methods=["POST"])
def feedback():
    # Expecting JSON like: {"image_id": "some-id", "feedback": "Positive"} or "Negative"
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON in request."}), 400

    feedback_value = data.get("feedback", "")
    image_id = data.get("image_id", "N/A")
    
    # Registrar feedback usando logging estruturado
    logger.info("Feedback recebido", extra={
        "event": "user_feedback",
        "image_id": image_id,
        "feedback": feedback_value
    })
    return jsonify({"mensagem": "Feedback recebido"}), 200

@app.route("/reload-model", methods=["POST"])
def reload_model():
    global model
    global model_thread
    try:
        logger.info("Recarregando modelo do MLflow", extra={
            "event": "model_reload_start"
        })
        model = None  # Resetar modelo
        # Start new background loading
        model_thread = threading.Thread(target=load_model_async, daemon=True)
        model_thread.start()
        logger.info("Recarregamento do modelo iniciado em background", extra={
            "event": "model_reload_started"
        })
        return jsonify({"mensagem": "Recarregamento do modelo iniciado em background"}), 200
    except Exception as e:
        logger.error("Erro ao recarregar modelo", extra={
            "event": "model_reload_error",
            "error": str(e)
        })
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
