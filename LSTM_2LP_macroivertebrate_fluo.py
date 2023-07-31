import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

data = pd.read_csv(r"../Ksenobiotici_makroinvertebrate.csv",encoding='latin-1')
X = data.iloc[2:,[79,80,64,76,89,88,83]].values  
y = data.iloc[2:,[170,164]].values 
X=X.astype("float32")
y=y.astype("float32")

## logging
##print(X)
print(X.shape)
##print (y)
print(y.shape)
print(len(y))

train_index=np.arange(0, X.shape[0])#train index svi indeksi
test_index= np.random.randint(0, X.shape[0],2)#nasumice uzima 10 za test
train_index=np.setdiff1d(train_index, test_index)# novi train indeks bez testa
## adding one more dimension to the dataset
X = X.reshape(len(X), 1, X.shape[1])
X_train=X[train_index,:]#sve vrednosti iz train seta
X_test=X[test_index,:]
y_train=y[train_index]
y_test=y[test_index]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt

# define the model
model = Sequential()
model.add(LSTM(10))
#model.add(Dense(8, activation='sigmoid'))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(2, activation='linear'))

tf.random.set_seed(12345)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
#learning rate 0.001

history=model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=1, validation_split = 0.2)

hist = pd.DataFrame(history.history)#beleži obuku po epohama
hist['epoch'] = history.epoch

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')

plt.xlabel('Epoch')
plt.ylabel('Error [Bromacil]')
plt.legend()
plt.grid(True)
