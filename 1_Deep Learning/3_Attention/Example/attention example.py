# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 20:21:50 2020

@author: Mee
"""

# Reference: https://yjam.tistory.com/72
# ===========================================================================

from keras import layers, Model
import pandas as pd
import numpy as np

input_dims = 32

def build_model(input_dims):
    # Input Layer
    input_layer = layers.Input(shape=(input_dims,))
    
    # Attention Layer
    attention_weights = layers.Dense(input_dims, activation='softmax')(input_layer)
    attention_weights = layers.Dense(input_dims, activation='softmax')(attention_weights)
    context_vector = layers.multiply([input_layer, attention_weights])
    
    # Full Conntected Layer: Attention vector
    fc_vector = layers.Dense(64)(context_vector)
    y = layers.Dense(1, activation='sigmoid')(fc_vector)
    
    return Model(inputs=[input_layer], outputs=y)

model = build_model(input_dims)
model.summary()

def get_data(n, input_dims, attention_column=1):
    train_x = np.random.standard_normal(size=(n, input_dims))
    train_y = np.random.randint(low=0, high=2, size=(n,1))
    train_x[:,attention_column]= train_y[:,0]
    
    return (train_x, train_y)


train_x, train_y = get_data(10000, 32, 5)
test_x, test_y = get_data(10000, 32, 5)

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=20, batch_size=64, validation_split=0.5, verbose=2)

layer_output = [layer.output for layer in model.layers]
layer_model = Model(inputs=model.input, outputs = layer_output)

layer_result = layer_model.predict(test_x)

attention_mean = np.mean(layer_result[2], axis=0)
df = pd.DataFrame(attention_mean.transpose(), columns=['attetion (%)'])
df.plot.bar()
