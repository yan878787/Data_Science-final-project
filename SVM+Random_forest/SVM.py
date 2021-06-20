# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 05:31:55 2021

@author: user
"""
import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

Rd_train = pd.read_csv('Rd_BTC_train.csv') 
Rd_test = pd.read_csv('Rd_BTC_test.csv') 

training_set = Rd_train.iloc[:, 1:2].values  # 取「Open」欄位值
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


X_rd_train = []   #預測點的前 44天的資料
y_rd_train = []   #預測點
for i in range(44, 1790):  #574 是訓練集總數
    X_rd_train.append(training_set_scaled[i-44:i, 0])
    y_rd_train.append(training_set_scaled[i, 0])
X_rd_train, y_rd_train = np.array(X_rd_train), np.array(y_rd_train,dtype=int)

real_bitcoin_price = Rd_test.iloc[:, 1:2].values
Rd_total = pd.concat((Rd_train ['Adj_Close_new'], Rd_test['Adj_Close_new']), axis = 0)
inputs = Rd_total[len(Rd_total) - len(Rd_test) - 44:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) # Feature Scaling 
X_rd_test = []
y_rd_test = []
for i in range(44, 201):  # timesteps一樣44； 88 = 先前的44天資料+2021年5,6月的44天資料
    X_rd_test.append(inputs[i-44:i, 0])
    y_rd_test.append(inputs[i, 0])
X_rd_test = np.array(X_rd_test)
y_rd_test = np.array(y_rd_test ,dtype=int)
X_rd_test = X_rd_test.reshape((X_rd_test.shape[0], X_rd_test.shape[1]))
X_rd_train = X_rd_train.reshape((X_rd_train.shape[0], X_rd_train.shape[1]))

svm = SVC(random_state=7)
svm.fit(X_rd_train,y_rd_train)
svm_predict = svm.predict(X_rd_test)

#evaluate

# accuracy score
acc = accuracy_score(y_rd_test, svm_predict)
print(f"Accuracy score: {round(acc,4)*100}%")

# precision score
precision = precision_score(y_rd_test, svm_predict,average='weighted')
print(f"Precision score: {round(precision,4)*100}%")

# recall_score
recall = recall_score(y_rd_test, svm_predict,average='weighted')
print(f"Recall score: {round(recall,4)*100}%")

# f1 score
f1 = f1_score(y_rd_test, svm_predict,average='weighted')
print(f"f1 score: {round(f1,4)*100}%")

#confusion matrix
svm_matrix = confusion_matrix(y_rd_test, svm_predict)
sns.heatmap(pd.DataFrame(svm_matrix), annot=True,cmap="Reds" , fmt='g')
plt.tight_layout()
plt.title('SVM', y=1.1)