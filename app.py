import logging

import mlflow
import pandas as pd
from dotenv import dotenv_values
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)
config = dotenv_values(".env")


def predict(input_value):
    """Obtain the model's inference from the given input."""
    logger.info(f"CONFIGS: {config}")
    TRACKING_SERVER_HOST = config.get(
        "TRACKING_SERVER_HOST", "test_tracking_server"
    )  # public DNS of the EC2 instance

    client = MlflowClient(tracking_uri=f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

    mlflow_model = client.search_registered_models()[0].latest_versions[0]
    run_id = mlflow_model.run_id

    model_metadata = {
        "name": mlflow_model.name,
        "run_id": run_id,
        "current_scope": mlflow_model.current_stage,
        "status": mlflow_model.status,
        "version": mlflow_model.version,
    }
    logged_model = f"s3://mlflow-artifact-remote-isa/3/{run_id}/artifacts/models/"

    model = mlflow.sklearn.load_model(logged_model)

    logger.info("Predicting: %s", str(input_value))

    if isinstance(input_value, dict):
        input_value = pd.DataFrame(input_value)

    predictions = model.predict_proba(input_value)[:, 1]
    response = {
        "model": model_metadata,
        "predictions": predictions.tolist(),
    }

    return jsonify(response)


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    logger.info(f"Route predict")
    features = request.get_json()
    logger.info(f"FEATURES: {features}")
    pred = predict(features)
    return pred


@app.route("/")
def hello_world():
    logger.info("Route hello world")
    logger.info(f"CONFIGS: {config}")
    return "Holi Dani! Te envío un abrazo digital"


if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=9696)
    app.run()
