"""Custom transformers module"""

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer to split a dataframe to keep only
    the provided columns
    """

    def __init__(self, feature_columns: list) -> None:
        self.feature_columns = feature_columns

    def transform(self, data):
        """Subslicing data"""
        return data[self.feature_columns].replace("", None)

    def fit(self, *_):
        """
        The "fit" function returns the object itself without
        any modifications for further use
        """
        return self


class ToNumeric(BaseEstimator, TransformerMixin):
    """
    The "ToNumeric" class transforms input data into a float data type using
    the "transform" function and returns the object itself without any
    modifications using the "fit" function.
    """

    def transform(self, data):
        """
        The function transforms the input data into a float data type.

        :param data: The input data that is being transformed. It is expected to be a
        pandas dataframe.
        :return: the input data as a df with data type "float".
        """
        return data.astype("float")

    def fit(self, *_):
        """
        The "fit" function returns the object itself without
        any modifications for further use
        """
        return self


class FillNA(BaseEstimator, TransformerMixin):
    """
    Transformer to replace NaN on the DataFrame using
    a default value
    """

    def __init__(self, default_value=0) -> None:
        self.default_value = default_value

    def transform(self, data):
        """
        This function fills missing values in a given dataset with a default value.

        :param data: The "data" parameter is a pandas DataFrame object that contains the data to be
        transformed.
        :return: The `transform` method is returning a modified version of the input
        `data` where any missing values (NaNs) have been filled with a default
        value specified by the `self.default_value` attribute.
        """
        return data.fillna(self.default_value)

    def fit(self, *_):
        """
        The "fit" function returns the object itself without
        any modifications for further use
        """
        return self


class ToDict(BaseEstimator, TransformerMixin):
    """
    Transformer to convert a DataFrame into a python dict
    """

    def transform(self, data) -> dict:
        """
        This function transforms a pandas DataFrame into a dictionary of records.

        :param data: data is a pandas DataFrame containing the input data
        that needs to be transformed
        :return: A dictionary representation of the input data, with each
        row of data being represented as a dictionary.
        """
        return data.to_dict(orient="records")

    def fit(self, *_):
        """
        The "fit" function returns the object itself without
        any modifications for further use
        """
        return self
