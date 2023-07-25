import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from s3_client import download_data, upload_data
from dotenv import dotenv_values
from settings import MODEL_PARAMETERS, OUTPUT_COLUMN, TEST_SIZE
import os
from datetime import date

from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier
from transformers import FeatureExtractor, FillNA, ToDict, ToNumeric
import mlflow

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

config = dotenv_values(".env")
logger = logging.getLogger(__name__)

TRACKING_SERVER_HOST = config.get(
    "TRACKING_SERVER_HOST", "test_tracking_server"
)  # public DNS of the EC2 instance


@task
def validate_data(data):
    """Validate the dataset's integrity."""
    if data.empty:
        raise Exception("Empty dataset")


@flow(name="Data generation", log_prints=True)
def generate_data():
    """Data extraction and transformation."""

    output_file = config.get("PATH_TO", "test_path_to")

    download_data(
        aws_key=config.get("AWS_KEY", "test_key"),
        aws_secret=config.get("AWS_SECRET", "test_secret"),
        bucket=config.get("BUCKET", "test_bucket"),
        prefix=config.get("PATH_FROM", "test_path_from"),
        output_name=output_file,
    )

    with open(output_file, "rb") as data_file:
        data = pickle.load(data_file)

    dataframe = pd.DataFrame(data)

    # Format output column
    target_column = MODEL_PARAMETERS.get("target_column")
    condition = MODEL_PARAMETERS.get("condition_value")

    dataframe[OUTPUT_COLUMN] = (dataframe[target_column] != condition).astype(int)

    train = []
    test = []

    try:
        # Separate dataset into train and test data
        train, test = train_test_split(dataframe, test_size=TEST_SIZE)
        validate_data.with_options(name="Train Dataset Validation").submit(train)
        validate_data.with_options(name="Test Dataset Validation").submit(train)
    except Exception as error:
        logger.error("The generated dataset is not valid: %s", str(error))
        raise

    if os.path.exists(output_file):
        os.remove(output_file)

    return train, test


class InvalidModelError(Exception):
    """Exception to raise when model performance is not
    the expeted"""


@task(name="Pipeline creation and fit", log_prints=True)
def create_model(
    dataset: pd.DataFrame, numeric_columns: list, categorical_columns: list, **kwargs
):
    """Train and test the model instance, from the given dataset."""

    output_column = OUTPUT_COLUMN
    model = None
    hyperparameters = kwargs.get("hyperparameters")

    numerical_features_pipeline = make_pipeline(
        FeatureExtractor(numeric_columns),
        ToNumeric(),
        FillNA(),
        StandardScaler().set_output(transform="pandas"),
    )

    categorical_features_pipeline = make_pipeline(
        FeatureExtractor(categorical_columns),
        SimpleImputer(missing_values=None, strategy="most_frequent").set_output(
            transform="pandas"
        ),
        ToDict(),
        DictVectorizer(sparse=False),
    )

    feature_union = FeatureUnion(
        [
            ("numerical_features", numerical_features_pipeline),
            ("categorical_features", categorical_features_pipeline),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("features_extraction", feature_union),
            ("classifier", XGBClassifier()),
        ]
    )

    model = GridSearchCV(
        pipeline,
        param_grid=hyperparameters,
        scoring="roc_auc",
        cv=5,
        verbose=1,
    )
    model.fit(dataset, dataset[output_column])
    return model


def predict(model, input_value):
    """Obtain the model's inference from the given input."""
    logger.info("Predicting: %s", str(input_value))

    if isinstance(input_value, dict):
        input_value = pd.DataFrame(input_value)

    return model.predict_proba(input_value)[:, 1]


@task(name="Model serialization", log_prints=True)
def serialize(model):
    """Serialize current model instance into a stream of bytes."""
    logger.info("Saving model as pickle...")
    model_pickle = pickle.dumps(model)
    with open("model.pkl", "wb") as file:
        pickle.dump(model_pickle, file, protocol=pickle.HIGHEST_PROTOCOL)


@task(name="Metrics calculation", log_prints=True)
def calculate_metrics(
    model, training_dataset: pd.DataFrame, test_dataset: pd.DataFrame, **kwargs
) -> dict:
    numeric_columns = kwargs.get("numeric_columns")
    categorical_columns = kwargs.get("categorical_columns")

    """Calculate roc auc score on training and test datsets"""
    train_predictions = predict(
        model, training_dataset[numeric_columns + categorical_columns]
    )
    test_predictions = predict(
        model, test_dataset[numeric_columns + categorical_columns]
    )

    train_auc = roc_auc_score(training_dataset[OUTPUT_COLUMN].values, train_predictions)
    test_auc = roc_auc_score(test_dataset[OUTPUT_COLUMN].values, test_predictions)

    model_metrics = {
        "roc_auc_score_training": train_auc,
        "roc_auc_score_test": test_auc,
    }

    markdown__auc_report = f"""# AUC ROC Report

    ## Summary

    Laundering money Prediction 

    ## ROC AUC XGBClassifier Model

    | Region    | AUC ROC Train | AUC ROC Test |
    |:----------|---------------|-------------:|
    | {date.today()} |     {train_auc:.2f}     |     {test_auc:.2f}     |
    """

    create_markdown_artifact(
        key="laundering-model-report", markdown=markdown__auc_report
    )

    return model_metrics


@flow(name="Train Laundering Money Model", log_prints=True)
def train_model(train_dataset, test_dataset):
    model = create_model(train_dataset, **MODEL_PARAMETERS)
    logger.info(
        "Trained MoneyLaunderingModel with params: %s",
        str(MODEL_PARAMETERS.get("hyperparameters")),
    )
    serialize(model)

    model_metrics = calculate_metrics(
        model, train_dataset, test_dataset, **MODEL_PARAMETERS
    )
    logger.info(
        "Model metrics: %s",
        str(model_metrics),
    )
    return model, model_metrics


@flow(name="Main training flow", log_prints=True)
def main_flow_training():
    """The main training pipeline"""

    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment("consumers-laundering-model")
    run_description = "Laundering model experiment description."
    tags = {
        "owner_team": "credits",
        "deployer": "mjaramillo",
        "responsible": "pepito",
        "execution_type": "live-scoring-model",
        "features": "path/to/features",
    }

    with mlflow.start_run(description=run_description, tags=tags):
        mlflow.log_params(MODEL_PARAMETERS)

        train_dataset, test_dataset = generate_data()
        model, model_metrics = train_model(train_dataset, test_dataset)

        mlflow.log_metric("train_roc_auc", model_metrics.get("roc_auc_score_training"))
        mlflow.log_metric("test_roc_auc", model_metrics.get("roc_auc_score_test"))

        mlflow.sklearn.log_model(model, artifact_path="models")
        print(f"Default artifacts URI: '{mlflow.get_artifact_uri()}'")

        upload_data(
            aws_key=config.get("AWS_KEY_MODELS", "test_key"),
            aws_secret=config.get("AWS_SECRET_MODELS", "test_secret"),
            bucket=config.get("MODELS_BUCKET", "test_bucket"),
            file_name=config.get("FILENAME", "test_file_name"),
            object_name=config.get("OBJECT_NAME", "test_object_name"),
        )


if __name__ == "__main__":
    main_flow_training()
