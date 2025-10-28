import joblib
from config import MAIN_MODEL
import pandas as pd


class ViolencePredictor:
    def __init__(self):
        self.model = joblib.load(MAIN_MODEL)

    def _preprocess_data_pdict(self, data: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [
            "video_name",
            "frame_index",
            "timestamp",
            "frame_width",
            "frame_height",
            "person1_id",
            "person2_id",
            "person1_idx",
            "person2_idx",
        ]
        data = data.drop(columns=cols_to_drop)
        return data

    def predict(self, data):
        data = self._preprocess_data_pdict(data)
        y_pred = self.model.predict(data)
        print(y_pred)

        return y_pred


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv("data/fight_train.csv")
    data = data[0:20]
    print("dataloaded")
    VP = ViolencePredictor()
    VP.predict(data)
