import numpy as np
import pandas as pd
import pandera.errors
import pytest

from transilien.data.schemas import features_schema, target_schema, validate_dataset

SCHEMA_ERRORS = (pandera.errors.SchemaError, pandera.errors.SchemaErrors)


def test_valid_frames_pass(toy_frame):
    x, y = toy_frame
    validate_dataset(x, y.to_frame(), x.copy())


def test_parasite_column_rejected(toy_frame):
    x, _ = toy_frame
    with pytest.raises(SCHEMA_ERRORS):
        features_schema().validate(x.assign(**{"Unnamed: 0": 0}))


def test_nan_rejected(toy_frame):
    x, _ = toy_frame
    bad = x.copy()
    bad.loc[0, "p0q2"] = np.nan
    with pytest.raises(SCHEMA_ERRORS):
        features_schema().validate(bad)


def test_weekend_rejected(toy_frame):
    x, _ = toy_frame
    bad = x.copy()
    bad.loc[0, "date"] = pd.Timestamp("2023-04-08")  # samedi
    with pytest.raises(SCHEMA_ERRORS):
        features_schema().validate(bad)


def test_unknown_station_rejected(toy_frame):
    x, y = toy_frame
    test = x.copy()
    test.loc[0, "gare"] = "ZZZ"
    with pytest.raises(SCHEMA_ERRORS):
        validate_dataset(x, y.to_frame(), test)


def test_non_integer_target_rejected(toy_frame):
    _, y = toy_frame
    with pytest.raises(SCHEMA_ERRORS):
        target_schema.validate((y + 0.5).to_frame())


def test_misaligned_rejected(toy_frame):
    x, y = toy_frame
    with pytest.raises(ValueError):
        validate_dataset(x, y.iloc[:-1].to_frame(), x)


def test_real_data_is_valid(dataset):
    validate_dataset(dataset.x_train, dataset.y_train.to_frame(), dataset.x_test)
    assert len(dataset.x_test) == 20657
