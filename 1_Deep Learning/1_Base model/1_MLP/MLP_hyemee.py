from keras.layers import Dense
from keras.models import Sequential
from keras.utils import plot_model

wd = 'AI_Study/1_Deep Learning/1_Base model/1_MLP/'
model_in = 5
model_out = 2
hidden = [32,32,32]

model = Sequential()

model.add(Dense(hidden[0], activation='relu', input_shape=(model_in,), name='Hidden-1'))
model.add(Dense(hidden[1], activation='relu', name='Hidden-2'))
model.add(Dense(hidden[2], activation='relu', name='Hidden-3'))
model.add(Dense(model_out, activation = 'softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy')

plot_model(model, show_shapes=True, to_file=f'{wd}model_hm.png')