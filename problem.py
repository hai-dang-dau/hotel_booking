import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from sklearn.metrics import precision_score, recall_score

problem_title = 'Hotel booking cancellation prediction'
_target_column_name = 'IsCanceled'
_prediction_label_names = [0, 1]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

class Precision(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='prec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, true, pred):
        return precision_score(true, pred)

class Recall(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='rec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, true, pred):
        return recall_score(true, pred)

class ModifiedF1(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='mf1', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, true, pred, w = 3):
        inv_prec = 1/precision_score(true, pred) # numpy automatically handles correctly any division by zero
        inv_rec = 1/recall_score(true, pred)
        inv_score = 1/(w + 1) * (inv_prec + w * inv_rec)
        return 1/inv_score

score_types = [ModifiedF1(), Precision(), Recall()]

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits = 8, test_size = 0.2)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
