import pandas as pd
class FeatureExtractor:
    def __init__(self):
        pass
    
    def fit(self, X_df, y):
        pass
    
    def transform(self, X_df):
        # simple solution: drop all numeric columns
        X_df_2 = X_df.fillna(-1)
        non_numeric_columns = ['ArrivalDateMonth', 'Meal', 'Country', 'MarketSegment', 'DistributionChannel', 'ReservedRoomType', 'AssignedRoomType', 'DepositType', 'Agent', 'Company', 'CustomerType']
        X_df_2 = X_df_2.drop(labels = non_numeric_columns, axis = 1)
        temp = X_df_2.values
        assert(str(temp.dtype) != 'object')
        return temp
