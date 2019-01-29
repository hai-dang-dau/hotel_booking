from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import numpy as np

class Classifier(BaseEstimator):
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression())
    
    def fit(self, X, y):
        
        # Train using traditional Gradient Boosting Decision Tree (of lightgbm package)
        # Train data set
        train_data = lgb.Dataset(X, label=y) 
        
        # Parameters
        param = {'boosting':'gbdt', 'num_leaves':31, 'objective':'binary',
            'learning_rate':.05, 'max_bin':255, 'min_data_in_leaf':100,
            'bagging_freq':1, 'bagging_fraction':1., 'lambda_l1':1, 'lambda_l2':1}
        param['metric'] = 'binary_logloss'
        
        num_round = 600
        
        # Train
        self.model = lgb.train(param, train_data, num_round)
        
    def predict_proba(self, X):
        pred = self.model.predict(X)
        return np.array([1 - pred, pred]).T