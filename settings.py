"""
General app's level configuration items.

"""

MODEL_HYPERPARAMETERS = {
    "classifier__n_estimators": [10, 5, 3, 2],
    "classifier__max_depth": [2, 3, 4, 5],
}

MODEL_PARAMETERS = {
    "numeric_columns": ["total_outcome_dollar_amount", "total_income_dollar_amount"],
    "categorical_columns": ["risk_pld"],
    "target_column": "conclusion",
    "condition_value": "CLOSED_NO_ROI",
    "hyperparameters": MODEL_HYPERPARAMETERS,
}
THE_ANSWER_TO_LIFE_THE_UNIVERSE_AND_EVERYTHING = 0.026
OUTPUT_COLUMN = "target"
TEST_SIZE = 0.3
DECIMALS = 8
