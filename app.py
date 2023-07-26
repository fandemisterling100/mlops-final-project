import logging
from dotenv import dotenv_values
import pandas as pd

import mlflow
from flask import Flask, request
from mlflow.tracking import MlflowClient


config = dotenv_values(".env")
logger = logging.getLogger(__name__)

TRACKING_SERVER_HOST = config.get(
    "TRACKING_SERVER_HOST", "test_tracking_server"
)  # public DNS of the EC2 instance

client = MlflowClient(tracking_uri=f"http://{TRACKING_SERVER_HOST}:5000")
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

RUN_ID = client.search_registered_models()[0].latest_versions[0].run_id
logged_model = f"s3://mlflow-artifact-remote-isa/3/{RUN_ID}/artifacts/models/"

model = mlflow.sklearn.load_model(logged_model)


def predict(input_value):
    """Obtain the model's inference from the given input."""
    logger.info("Predicting: %s", str(input_value))

    if isinstance(input_value, dict):
        input_value = pd.DataFrame(input_value)

    predictions = model.predict_proba(input_value)[:, 1]

    return predictions.tolist()


app = Flask("laundering-model-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    features = request.get_json()
    pred = predict(features)
    return pred


if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0', port=9696)
    app.run()
