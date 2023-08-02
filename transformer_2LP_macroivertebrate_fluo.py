import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import layers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GlobalAveragePooling1D
from keras.layers import Input
import matplotlib.pyplot as plt

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

data = pd.read_csv(r"../Ksenobiotici_makroinvertebrate.csv",encoding='latin-1')
## Fluoranthene & 2.4 D 
#X = data.iloc[2:,[79,80,64,76,89,88,83]].values  
#y = data.iloc[2:,[170,164]].values 
## Bromacil
X = data.iloc[2:,[73,69,68,70,72,58,60,91]].values  
y = data.iloc[2:,[166]].values 
## Bentazone
#X = data.iloc[2:,[64,45,89,88,83,107,90,93,91]].values  
#y = data.iloc[2:,[169]].values 

X=X.astype("float32")
y=y.astype("float32")

## logging
##print(X)
print(X.shape)
##print (y)
print(y.shape)
print(len(y))


train_index=np.arange(0, X.shape[0])#train index all indexes
test_index= np.random.randint(0, X.shape[0],2)#nasumice uzima 10 za test
train_index=np.setdiff1d(train_index, test_index)# novi train indeks bez testa

## adding one more dimension to the dataset for LSTM
X = X.reshape(len(X), 1, X.shape[1])
X_train=X[train_index,:]#sve vrednosti iz train seta
X_test=X[test_index,:]
y_train=y[train_index]
y_test=y[test_index]

vocab_size = 10 # 20000  # Only consider the top 20k words
maxlen = 20 # Only consider the first 200 words 
embed_dim = 22 # 32  # Embedding size for each token
num_heads = 4 # 10  # Number of attention heads
ff_dim = 32 # 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)


# define the model
model = Sequential()
inputs = Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
#x = GlobalAveragePooling1D()(x)
#model.add(Dense(8, activation='sigmoid'))
#model.add(Dense(12, activation='sigmoid'))
#model.add(Dense(30, activation='sigmoid'))
model.add(Dense(2, activation='linear'))

tf.random.set_seed(12345)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
#learning rate 0.001
## T(10) => MSE = 0.0172
## T(05) => MSE = 3.6691e-05
## T(03) => MSE = 4.1896e-05
## T(04) => MSE = 3.9311e-05

history=model.fit(X_train, y_train, epochs=2000, batch_size=10, verbose=1, validation_split = 0.2)

hist = pd.DataFrame(history.history)#beleži obuku po epohama
hist['epoch'] = history.epoch

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')

plt.xlabel('Epoch')
plt.ylabel('Error [Bromacil]')
plt.legend()
plt.grid(True)
