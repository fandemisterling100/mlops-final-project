"""Module to generate evidently reports to show data on Grafana"""

import datetime

import psycopg
from dotenv import dotenv_values
from evidently import ColumnMapping
from evidently.metric_preset import DataQualityPreset
from evidently.report import Report
from prefect import flow, task

from settings import MODEL_PARAMETERS, OUTPUT_COLUMN

config = dotenv_values(".env")
DB_HOST = config.get("DB_HOST", "127.0.0.1")

CREATE_TABLE_STATEMENT = """
create table if not exists metrics_summary(
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
    current_roc_auc float,
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
    reference_unique_risk_pld integer,
    reference_roc_auc float
)
"""
num_features = MODEL_PARAMETERS.get("numeric_columns")
cat_features = MODEL_PARAMETERS.get("categorical_columns")


@task(name="Prepare DB to save report", log_prints=True)
def prep_db():
    """
    The function `prep_db` checks if a database named "test" exists,
    and if not, creates it, and then connects to the "test"
    database and executes a create table statement.
    """
    with psycopg.connect(
        f"host={DB_HOST} port=5432 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect(
            f"host={DB_HOST} port=5432 dbname=test user=postgres password=example"
        ) as conn:
            conn.execute(CREATE_TABLE_STATEMENT)


@task(name="Run report", log_prints=True)
def check_data_quality(ref, curr):
    """
    The function `check_data_quality` compares the data
    quality between two datasets using a set of
    predefined metrics.

    :param ref: The `ref` parameter is the reference dataset
    that you want to compare the current dataset (`curr`) against.
    Both `ref` and `curr` should be pandas DataFrames
    :param curr: The `curr` parameter represents the current
    dataset that you want to check the data
    quality for. It should be a pandas DataFrame containing the data
    :return: a data quality report.
    """
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


@task(name="Save report", log_prints=True)
def save_data_quality_report(result, train_roc_auc, test_roc_auc, run_id):
    """
    The `save_data_quality_report` function saves data quality
    metrics and other information to a PostgreSQL database.

    :param result: The `result` parameter is a dictionary that
    contains the metrics and results of the data quality report.
    It should have the following structure:
    :param train_roc_auc: The `train_roc_auc` parameter is
    the ROC AUC (Receiver Operating Characteristic Area Under
    the Curve) score for the training data. It is a measure of
    the model's performance in predicting the target variable
    for the training data
    :param test_roc_auc: The `test_roc_auc` parameter is the
    ROC AUC (Receiver Operating Characteristic
    Area Under the Curve) score for the test dataset.
    It is a performance metric used to evaluate the
    quality of a classification model
    :param run_id: The `run_id` parameter is used to
    identify the specific run or iteration of the data
    quality report. It can be any unique identifier that
    helps distinguish one report from another
    """
    metrics = result.get("metrics")
    current_metrics = metrics[0].get("result").get("current")
    reference_metrics = metrics[0].get("result").get("reference")

    with psycopg.connect(
        f"host={DB_HOST} port=5432 dbname=test user=postgres password=example",
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
                    current_roc_auc,
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
                    reference_unique_risk_pld,
                    reference_roc_auc) values (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
                    train_roc_auc,
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
                    test_roc_auc,
                ),
            )


@flow(name="Generate Data Quality Report", log_prints=True)
def generate_data_quality_report(ref, curr, train_roc_auc, test_roc_auc, run_id):
    """
    The function generates a data quality report and saves
    it along with other metrics and a run ID.
    """
    report = check_data_quality(ref, curr)
    save_data_quality_report(report.as_dict(), train_roc_auc, test_roc_auc, run_id)


@flow(name="Generate evidently reports", log_prints=True)
def generate_evidently_reports(
    train_dataset, test_dataset, train_roc_auc, test_roc_auc, run_id
):
    """
    The function `generate_evidently_reports` generates data
    quality reports for the train and test datasets, along with
    their respective ROC AUC scores, and associates them with a
    specific run ID.
    """
    prep_db()
    generate_data_quality_report(
        train_dataset, test_dataset, train_roc_auc, test_roc_auc, run_id
    )
