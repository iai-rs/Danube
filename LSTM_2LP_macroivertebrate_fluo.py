import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = pd.read_csv(r"../Ksenobiotici_makroinvertebrate.csv",encoding='latin-1')
#print("Bromacil")
#X = data.iloc[2:,[73,69,68,70,72,58,60,91]].values  
#y = data.iloc[2:,[166]].values 
#print("2.4 D")
#X = data.iloc[2:,[64,55,93,111,116,35,161,162]].values  
#y = data.iloc[2:,[164]].values 
#print("Fluoranthene")
#X = data.iloc[2:,[79,80,64,76,89,88,83]].values  
#y = data.iloc[2:,[170]].values 
print("Fluoranthene & 2.4 D")
X = data.iloc[2:,[79,80,64,76,89,88,83]].values  
y = data.iloc[2:,[170,164]].values 
#print("Bentazone")
#X = data.iloc[2:,[64,45,89,88,83,107,90,93,91]].values  
#y = data.iloc[2:,[169]].values 

X=X.astype("float32")
y=y.astype("float32")

## logging
##print(X)
#print(X.shape)
##print (y)
#print(y.shape)
#print(len(y))

train_index=np.arange(0, X.shape[0])#train index all indexes
test_index= np.random.randint(0, X.shape[0],2)#random 10 for test
train_index=np.setdiff1d(train_index, test_index)# new train index no test
## adding one more dimension to the dataset
# X = X.reshape(len(X), 1, X.shape[1])

X_train=X[train_index,:]#all values from train set
X_test=X[test_index,:]
y_train=y[train_index]
y_test=y[test_index]

X_train = X_train.reshape(len(X_train), 1, X_train.shape[1])

# define the model
model = Sequential()
model.add(LSTM(2))
#model.add(Dense(8, activation='sigmoid'))
#model.add(Dense(12, activation='sigmoid'))
#model.add(Dense(30, activation='sigmoid'))
model.add(Dense(2, activation='linear'))

tf.random.set_seed(12345)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
#learning rate 0.001
### Fluoro + D
## LSTM(20) => MSE = 0.0025
## LSTM(30) => MSE = 0.0045
## LSTM(10) => MSE = 0.0010
## LSTM(05) => MSE = 1.1112e-04
## LSTM(04) => MSE = 1.3198e-04
## LSTM(03) => MSE = 0.0013
## LSTM(02) => MSE = 2.5672e-05
## LSTM(01) => MSE = 0.0067
### Bromacil 
##MSE = 7.3138e-04

history=model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=1, validation_split = 0.2)

hist = pd.DataFrame(history.history)#beleži obuku po epohama
hist['epoch'] = history.epoch

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')

plt.xlabel('Epoch')
plt.ylabel('Error [Bromacil]')
plt.legend()
plt.grid(True)

## MSE
import numpy as np
from sklearn.metrics import mean_squared_error
# assume model and test set data are already defined
# predict output on test set
X_test = X_test.reshape(len(X_test), 1, X_test.shape[1])
y_pred = model.predict(X_test)
# calculate MSE
mse = mean_squared_error(y_test, y_pred)
# print the result
print("MSE=", mse)

## Bromacil
# 0.001426385
# 0.0031136826
# 0.0017592615
## 2.4 D
# 1.36419485e-05
# 3.263747e-05
# 8.033701e-06
## Fluoranthene
# 0.0077476795
# 8.323929e-08
# 0.0037221021
## Bentazone
# 0.00057681283
# 3.9730803e-06
# 1.6001328e-05
## Fluoranthene & 2.4 D
# 3.3694337e-07
# 3.5156765e-05
# 0.0001348218

