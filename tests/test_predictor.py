from backend.services.prediction.predictor import ViolencePredictor
import pandas as pd
import numpy as np

TEST_DATA = pd.read_csv("data/fight_train.csv")[0:20]


def test_model():
    VP = ViolencePredictor()
    assert np.array_equal(VP.predict(TEST_DATA), np.array([1] * 20))
