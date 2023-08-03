"""Module to manually test evidently report creation"""

import datetime
import os
import pickle

import pandas as pd
import psycopg
from dotenv import dotenv_values
from evidently import ColumnMapping
from evidently.metric_preset import DataQualityPreset
from evidently.report import Report
from sklearn.model_selection import train_test_split

from s3_client import download_data
from settings import MODEL_PARAMETERS, OUTPUT_COLUMN, TEST_SIZE

config = dotenv_values(".env")

CREATE_TABLE_STATEMENT = """
drop table if exists metrics_summary;
create table metrics_summary(
	timestamp timestamp,
    run_id varchar(256),
	current_number_cols integer,
    current_number_rows integer,
    current_number_missing_values integer,
    current_number_duplicated_rows integer,
    current_nan_total_outcome_dollar_amount integer,
    current_nan_total_income_dollar_amount integer,
    current_nan_risk_pld integer,
    current_unique_target integer,
    current_unique_total_outcome_dollar_amount integer,
    current_unique_total_income_dollar_amount integer,
    current_unique_risk_pld integer,
    reference_number_cols integer,
    reference_number_rows integer,
    reference_number_missing_values integer,
    reference_number_duplicated_rows integer,
    reference_nan_total_outcome_dollar_amount integer,
    reference_nan_total_income_dollar_amount integer,
    reference_nan_risk_pld integer,
    reference_unique_target integer,
    reference_unique_total_outcome_dollar_amount integer,
    reference_unique_total_income_dollar_amount integer,
    reference_unique_risk_pld integer
)
"""


# @task
def prep_db():
    """
    The function `prep_db` checks if a database named "test"
    exists, and if not, creates it, and then connects to the
    "test" database and executes a create table statement.
    """
    with psycopg.connect(
        "host=127.0.0.1 port=5432 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect(
            "host=127.0.0.1 port=5432 dbname=test user=postgres password=example"
        ) as conn:
            conn.execute(CREATE_TABLE_STATEMENT)


# @task
def check_data_quality(ref, curr):
    """
    The function `check_data_quality` takes in two dataframes,
    `ref` and `curr`, and performs data quality checks on
    specific columns using a predefined set of metrics.

    :param ref: The `ref` parameter is a reference dataset
    that contains the expected values for the
    features used in the model
    :param curr: The variable `curr` represents the current
    dataset that you want to check the data
    quality for
    :return: a data quality report.
    """
    num_features = MODEL_PARAMETERS.get("numeric_columns")
    cat_features = MODEL_PARAMETERS.get("categorical_columns")

    all_columns = [OUTPUT_COLUMN] + num_features + cat_features

    ref = ref[all_columns].replace("", None)
    curr = curr[all_columns].replace("", None)

    data_quality_report = Report(
        metrics=[
            DataQualityPreset(),
        ]
    )
    column_mapping = ColumnMapping(
        prediction=OUTPUT_COLUMN,
        numerical_features=num_features,
        categorical_features=cat_features,
        target=None,
    )
    data_quality_report.run(
        reference_data=ref, current_data=curr, column_mapping=column_mapping
    )

    return data_quality_report


def save_data_quality_report(result, run_id):
    """
    The `save_data_quality_report` function saves data quality
    metrics to a PostgreSQL database.

    :param result: The `result` parameter is a dictionary that
    contains the metrics for the data quality
    report. It has the following structure:
    :param run_id: The `run_id` parameter is an identifier
    for the data quality report run. It could be
    a unique identifier such as a UUID or an integer
    value that helps identify the specific run of the
    data quality report
    """
    metrics = result.get("metrics")
    current_metrics = metrics[0].get("result").get("current")
    reference_metrics = metrics[0].get("result").get("reference")

    with psycopg.connect(
        "host=127.0.0.1 port=5432 dbname=test user=postgres password=example",
        autocommit=True,
    ) as conn:
        with conn.cursor() as curr:
            curr.execute(
                """insert into metrics_summary(
                    timestamp,
                    run_id,
                    current_number_cols,
                    current_number_rows,
                    current_number_missing_values,
                    current_number_duplicated_rows,
                    current_nan_total_outcome_dollar_amount,
                    current_nan_total_income_dollar_amount,
                    current_nan_risk_pld,
                    current_unique_target,
                    current_unique_total_outcome_dollar_amount,
                    current_unique_total_income_dollar_amount,
                    current_unique_risk_pld,
                    reference_number_cols,
                    reference_number_rows,
                    reference_number_missing_values,
                    reference_number_duplicated_rows,
                    reference_nan_total_outcome_dollar_amount,
                    reference_nan_total_income_dollar_amount,
                    reference_nan_risk_pld,
                    reference_unique_target,
                    reference_unique_total_outcome_dollar_amount,
                    reference_unique_total_income_dollar_amount,
                    reference_unique_risk_pld) values (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )""",
                (
                    datetime.datetime.now(),
                    run_id,
                    current_metrics.get("number_of_columns"),
                    current_metrics.get("number_of_rows"),
                    current_metrics.get("number_of_missing_values"),
                    current_metrics.get("number_of_duplicated_rows"),
                    current_metrics.get("nans_by_columns").get(
                        "total_outcome_dollar_amount"
                    ),
                    current_metrics.get("nans_by_columns").get(
                        "total_income_dollar_amount"
                    ),
                    current_metrics.get("nans_by_columns").get("risk_pld"),
                    current_metrics.get("number_uniques_by_columns").get("target"),
                    current_metrics.get("number_uniques_by_columns").get(
                        "total_outcome_dollar_amount"
                    ),
                    current_metrics.get("number_uniques_by_columns").get(
                        "total_income_dollar_amount"
                    ),
                    current_metrics.get("number_uniques_by_columns").get("risk_pld"),
                    reference_metrics.get("number_of_columns"),
                    reference_metrics.get("number_of_rows"),
                    reference_metrics.get("number_of_missing_values"),
                    reference_metrics.get("number_of_duplicated_rows"),
                    reference_metrics.get("nans_by_columns").get(
                        "total_outcome_dollar_amount"
                    ),
                    reference_metrics.get("nans_by_columns").get(
                        "total_income_dollar_amount"
                    ),
                    reference_metrics.get("nans_by_columns").get("risk_pld"),
                    reference_metrics.get("number_uniques_by_columns").get("target"),
                    reference_metrics.get("number_uniques_by_columns").get(
                        "total_outcome_dollar_amount"
                    ),
                    reference_metrics.get("number_uniques_by_columns").get(
                        "total_income_dollar_amount"
                    ),
                    reference_metrics.get("number_uniques_by_columns").get("risk_pld"),
                ),
            )


def generate_data_quality_report(ref, curr, run_id):
    """
    The function `generate_data_quality_report` generates a
    data quality report by checking the data
    quality between two datasets and saves the
    report using a given run ID.
    """
    report = check_data_quality(ref, curr)
    save_data_quality_report(report.as_dict(), run_id)


# @flow(name="test evidently reports", log_prints=True)
def generate_reports():
    """
    The function `generate_reports` downloads data,
    processes it, separates it into train and test
    datasets, and generates a data quality report.
    """
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

    generate_data_quality_report(train, test, "fdsfsdf233434")


if __name__ == "__main__":
    prep_db()
    generate_reports()
