import logging
from dotenv import dotenv_values
import pandas as pd

import mlflow
from flask import Flask, request, jsonify
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

    RUN_ID = client.search_registered_models()[0].latest_versions[0].run_id
    logged_model = f"s3://mlflow-artifact-remote-isa/3/{RUN_ID}/artifacts/models/"

    model = mlflow.sklearn.load_model(logged_model)

    logger.info("Predicting: %s", str(input_value))

    if isinstance(input_value, dict):
        input_value = pd.DataFrame(input_value)

    predictions = model.predict_proba(input_value)[:, 1]
    response = {
        "run_id": RUN_ID,
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
