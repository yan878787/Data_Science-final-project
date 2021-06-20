# Import the libraries
import numpy as np
import matplotlib.pyplot as plt  # for 畫圖用
import pandas as pd

# Import the training set
dataset_train = pd.read_csv('Train.csv') # 讀取訓練集
training_set = dataset_train.iloc[:, 1:2].values  

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []   #預測點的前 44天的資料
y_train = []   #預測點
for i in range(44, 1790):  #1790 是訓練集總數
    X_train.append(training_set_scaled[i-44:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 200, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 200, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 200, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 200))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# 進行LSTM訓練
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

dataset_test = pd.read_csv('Test.csv')
real_bitcoin_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Adj Close'], dataset_test['Adj Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 44:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) # Feature Scaling

X_test = []
for i in range(44, 200):  # timesteps一樣44； 88 = 先前的44天資料+2021年5,6月的44天資料
    X_test.append(inputs[i-44:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshape 成 3-dimension

predicted_bitcoin_price = model.predict(X_test)
predicted_bitcoin_price = sc.inverse_transform(predicted_bitcoin_price)  # to get the original scale

prediction = np.array([])
for k in range(len(predicted_bitcoin_price)-1):
    prediction = np.append(prediction,predicted_bitcoin_price[k+1]-predicted_bitcoin_price[k])
for j in range(len(prediction)):
    if prediction[j]>0:
        prediction[j]=1
    else:
        prediction[j]=0

# Visualising the results
plt.plot(real_bitcoin_price, color = 'red', label = 'Real Bitcoin Price')  # 紅線表示真實股價
plt.plot(predicted_bitcoin_price, color = 'blue', label = 'Predicted Bitcoin Price')  # 藍線表示預測股價
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()

df3=pd.DataFrame(dataset_test[['Adj Close']])
df3[df3 <= 0] = 0		  #將小於等於0的資料改為0	
df3[df3 > 0 ]=1	
Y_test=df3[0:156]
ytest=df3[1:156]

scores = model.evaluate(X_test, Y_test) 
print(scores) 
print("\t[Info] LSTM's Accuracy of testing data = {:2.1f}%".format(scores*100.0))	#顯示正確率

from sklearn.metrics import confusion_matrix
conf_mx=confusion_matrix(ytest, prediction)      #帶入正確率套件
print(conf_mx)

tp=conf_mx[0,0]
fn=conf_mx[0,1]
fp=conf_mx[1,0]
tn=conf_mx[1,1]

print('Precision',tp/(tp+fp))
print('recall  ',tp/(tp+fn))