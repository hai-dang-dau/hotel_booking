import pandas as pd
class FeatureExtractor:
    def __init__(self):
        pass
    
    def fit(self, X_df, y):
        pass
    
    def transform(self, X_df):
        # Handle mising data
        X_df_2 = X_df.fillna(-1)
        
        # Transform categorical features into pandas's categorical type
        non_numeric_columns = ['ArrivalDateMonth', 'Meal', 'Country', 'MarketSegment', 'DistributionChannel', 
                               'ReservedRoomType', 'AssignedRoomType', 'DepositType', 'Agent', 'Company', 
                               'CustomerType']
        X_df_2[non_numeric_columns] = X_df_2[non_numeric_columns].astype('category')
        
        return X_df_2