import os
import numpy as np
from sklearn.externals import joblib
from keras.models import load_model


class SentimentClassificationModel:

    _label_mapping = {
        0: "negative",
        1: "neutral",
        2: "positive",
        3: "skip",
        4: "speech"
    }

    def __init__(self, joblib_pipeline_path):
        if not joblib_pipeline_path:
            raise ValueError("Provide a valid path to serialized pipeline")
        self.sklearn_pipeline = joblib.load(joblib_pipeline_path)

    def predict_sentiment(self, x):
        return self._encoded_prediction_to_label(self.sklearn_pipeline.predict(x))

    @classmethod
    def _encoded_prediction_to_label(cls, prediction):
        return np.array([cls._label_mapping[y_pred] for y_pred in prediction])


class KerasSentimentClassificationModel(SentimentClassificationModel):

    def __init__(self, model_weights_path, joblib_pipeline_path=None):
        if not model_weights_path:
            raise ValueError("Provide a valid path to model weights")
        if joblib_pipeline_path:
            super().__init__(joblib_pipeline_path)
        self.keras_model = load_model(model_weights_path)

    def predict_sentiment(self, x):
        preprocessed_data = self.sklearn_pipeline.transform(x) if self.sklearn_pipeline else x
        return self._encoded_prediction_to_label(self.keras_model.predict_classes(preprocessed_data))


def initialize_model(model_dir):
    model_dir_contents = os.listdir(model_dir)
    joblib_pipeline_paths = [s for s in model_dir_contents if s.endswith(".joblib")]
    keras_model_weights_paths = [s for s in model_dir_contents if s.endswith(".h5")]

    if len(joblib_pipeline_paths) > 1 or len(keras_model_weights_paths) > 1:
        raise ValueError("Ambigious model configuration, there should be at most 1 joblib and at most 1 h5 file")

    if len(keras_model_weights_paths) == 0:
        if len(joblib_pipeline_paths) == 1:
            return SentimentClassificationModel(os.path.join(model_dir, joblib_pipeline_paths[0]))
        else:
            raise ValueError("Model unspecified, there should be at least 1 joblib or h5 file")
    else:
        return KerasSentimentClassificationModel(os.path.join(model_dir, keras_model_weights_paths[0]),
                                                 os.path.join(model_dir, joblib_pipeline_paths[0])
                                                 if len(joblib_pipeline_paths) == 1 else None)
