# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 21:54:34 2021

@author: user
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

dataset_train = pd.read_csv('BTC_Train.csv')  # 讀取訓練集

np.random.seed(0)
#Y = df['Bankrupt']
array = dataset_train.values
X = array[:,1:93]
Y = array[:,0]
validation_size = 0.20
seed = 0
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'

rfc=RandomForestClassifier(n_estimators=250,n_jobs = -1,random_state =seed, min_samples_leaf = 50,max_features=7)
rfc.fit(X_train,Y_train)
importances = rfc.feature_importances_ 
labels=dataset_train.columns
sorted_indices = np.argsort(importances)[::-1]

print("重要性：",importances)

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f +1, 30, labels[sorted_indices[f]], importances[sorted_indices[f]]))


print('finish')