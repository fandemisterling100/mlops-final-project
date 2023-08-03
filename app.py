"""Flask app to serve ML model"""

import logging

import mlflow
import pandas as pd
from dotenv import dotenv_values
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

from settings import EXPERIMENT_NAME

logger = logging.getLogger(__name__)
config = dotenv_values(".env")

TRACKING_SERVER_HOST = config.get(
    "TRACKING_SERVER_HOST", "test_tracking_server"
)  # public DNS of the EC2 instance


def predict(input_value):
    """
    The `predict` function loads a trained ML model, makes predictions on
    the given input, and returns the predictions along with model metadata.

    :param input_value: The `input_value` parameter is the input data that you
    want to make predictions on. It can be either a dictionary or a pandas
    DataFrame. If it is a dictionary, it will be converted
    to a DataFrame before making predictions. The model will
    then use this input data to generate
    predictions using the `
    :return: a JSON response containing the model metadata and the predictions.
    """

    logger.info(f"CONFIGS: {config}")

    client = MlflowClient(tracking_uri=f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    stage = "Production"
    mlflow_models = client.search_model_versions(
        filter_string=f"name = '{EXPERIMENT_NAME}'", order_by=["version_number DESC"]
    )
    mlflow_model = mlflow_models[0]

    for model in mlflow_models:
        if model.current_stage == stage:
            mlflow_model = model

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
    """
    The function `predict_endpoint` takes in a JSON object of features,
    logs the features, and returns the prediction made using the
    `predict` function.
    :return: the prediction made by the `predict` function.
    """
    features = request.get_json()
    logger.info(f"Features: {features}")
    pred = predict(features)
    return pred


@app.route("/")
def hello_world():
    """
    The function `hello_world` returns the string
    "Hello world!" when the root URL is accessed.
    :return: The string "Hello world!" is being returned.
    """
    return "Hello world!"


if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=9696)
    app.run()
