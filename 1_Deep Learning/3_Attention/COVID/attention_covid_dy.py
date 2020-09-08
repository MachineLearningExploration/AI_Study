#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    get_ipython().run_line_magic('reload_ext', 'lab_black')
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception as e:
    print(e)

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("./multiTimeline.csv", header=1, index_col=0)
df


# In[3]:


df = df.rename(columns={df.columns[0]: "y"})


# In[4]:


def shift_data(data, offset):
    df = data.copy(deep=True)
    for i in range(offset):
        df.insert(len(df.columns), f"x{i+1}", df.iloc[:, -1].shift(1, fill_value=0))
    return df.iloc[offset:]


# In[5]:


shift_offset = 3
shifted_df = shift_data(df, shift_offset)
shifted_df


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(
    shifted_df.iloc[:, 1:], shifted_df["y"], test_size=0.2, shuffle=False
)
X_train.info(), X_test.info()


# In[7]:


def build_model(input_dim):
    input_layer = keras.layers.Input(shape=(input_dim,))

    attention_probs = keras.layers.Dense(input_dim, activation="softmax")(input_layer)
    attention_mul = keras.layers.multiply([input_layer, attention_probs])

    fc_attention_mul = keras.layers.Dense(64)(attention_mul)
    y = keras.layers.Dense(1, activation="sigmoid")(fc_attention_mul)

    return keras.Model(inputs=[input_layer], outputs=y)


# In[8]:


model = build_model(shift_offset)
model.summary()


# In[9]:


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=100, verbose=0)


# In[10]:


layer_outputs = [layer.output for layer in model.layers]
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

output_data = activation_model.predict(X_test)
len(output_data)


# In[11]:


attention_vector = np.mean(output_data[1], axis=0)
attention_vector


# In[12]:


df = pd.DataFrame(attention_vector.transpose(), columns=["attention (%)"])
df.plot.bar()

