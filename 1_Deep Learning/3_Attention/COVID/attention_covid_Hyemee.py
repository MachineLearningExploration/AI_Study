# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:28:16 2020

@author: Mee
"""

import pandas as pd
import numpy as np
from keras import layers, Model, optimizers
from sklearn.preprocessing import MinMaxScaler


# data load

data = pd.read_csv('covid_google_trend.csv')

scaler = MinMaxScaler()
tmp = pd.DataFrame(data.iloc[:,1])
data.iloc[:,1] = scaler.fit_transform(tmp)
del tmp

# parametets

ts = 14


# data preprocessing ####
    
def data_prep(val,n):

    df_len = len(val) - n
    
    for i in range(n):
        if i == 0:
            x = pd.DataFrame(val[i:(df_len+i-1)])
        else:
            tmp = val[i:(df_len+i-1)]
            tmp.index = x.index
            x = pd.concat([x,tmp], axis=1)
            del tmp
    
    y = val[(n+1):(df_len+n+1)]
    
    return (np.asarray(x),np.asarray(y))


train = data[data['Day'] < '2020-08-01']
test = data[data['Day'] >= '2020-08-01']
train_val = train.iloc[:,1]
test_val = test.iloc[:,1]

train_x, train_y = data_prep(train_val,ts)
test_x, test_y = data_prep(test_val,ts)


# attention modeling


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

model = build_model(ts)
model.summary()

model.compile(optimizer=optimizers.Adam(lr = 0.01),loss='mse')
model.fit(train_x, train_y, epochs=1000, verbose=2)

layer_output = [layer.output for layer in model.layers]
layer_model = Model(inputs=model.input, outputs = layer_output)

layer_result = layer_model.predict(test_x)

attention_mean = np.mean(layer_result[2], axis=0)
df = pd.DataFrame(attention_mean.transpose(), columns=['attetion (%)'])

print(np.where(df.iloc[:,0]==max(df.iloc[:,0])))
