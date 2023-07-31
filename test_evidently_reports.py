import logging
import os
import pickle
from datetime import date

import mlflow
import pandas as pd
from dotenv import dotenv_values
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset, DataQualityPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier

from s3_client import download_data
from settings import MODEL_PARAMETERS, OUTPUT_COLUMN, TEST_SIZE
from transformers import FeatureExtractor, FillNA, ToDict, ToNumeric

config = dotenv_values(".env")


@task
def check_data_quality(ref, curr):
    num_features = MODEL_PARAMETERS.get("numeric_columns")
    cat_features = MODEL_PARAMETERS.get("categorical_columns")

    column_mapping = ColumnMapping(
        prediction=OUTPUT_COLUMN,
        numerical_features=num_features,
        categorical_features=cat_features,
        target=None,
    )

    data_quality_report = Report(
        metrics=[
            DataQualityPreset(),
        ]
    )

    data_quality_report.run(
        reference_data=ref, current_data=curr, column_mapping=column_mapping
    )

    return data_quality_report


@task
def check_classification_performance(reference, predictions):
    classification_performance_report = Report(
        metrics=[
            ClassificationPreset(),
        ]
    )
    classification_performance_report.run(
        reference_data=reference, current_data=predictions
    )

    return classification_performance_report


@flow(name="test evidently reports", log_prints=True)
def generate_reports():
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
    except Exception as error:
        print(error)

    if os.path.exists(output_file):
        os.remove(output_file)
    # import ipdb; ipdb.set_trace()
    quality_report = check_data_quality(train, test).as_dict()
    print(quality_report)


if __name__ == "__main__":
    generate_reports()
