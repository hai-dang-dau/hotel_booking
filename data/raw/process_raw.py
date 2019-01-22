import pandas as pd
import numpy as np

np.random.seed(123)
h1 = pd.read_csv("H1.csv")
h1 = h1.drop(labels = ['ReservationStatus', 'ReservationStatusDate', 'ArrivalDateYear'], axis = 1)

p = np.random.permutation(len(h1))
sep = int(len(h1) * 0.8)
train_indices = p[0:sep]
test_indices = p[sep:]

train = h1.iloc[train_indices,:]
test = h1.iloc[test_indices,:]
train.to_csv("../train.csv", index = False)
test.to_csv("../test.csv", index = False)