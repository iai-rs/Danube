# Transformer attempt to be  used over Danube data

import logging

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Assuming you have the dataset as sparse matrices: X_train_sparse, y_train_dense
# Convert sparse matrices to dense arrays (if necessary)
## read xls file
X_train_dense = X_train_sparse.toarray()
# Define the Transformer model
def create_transformer_model(max_sequence_length):
    input_layer = layers.Input(shape=(max_sequence_length,))
    transformer_layer = layers.Transformer(
        num_heads=8,
        feed_forward_dim=512,
        dropout=0.2,
        activation="relu"
    )(input_layer)
    output_layer = layers.Dense(1)(transformer_layer)  # Single output for regression
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Define the model’s input shape
input_shape = X_train_dense.shape[1]
# Create the Transformer model
transformer_model = create_transformer_model(input_shape)
# Compile the model with mean squared error (MSE) as the loss function for regression
transformer_model.compile(
    loss="mean_squared_error",
    optimizer="adam"
)
# Train the model
transformer_model.fit(
    X_train_dense,
    y_train_dense,
    batch_size=64,
    epochs=10,
    validation_split=0.1
)
