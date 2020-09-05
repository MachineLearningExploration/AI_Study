from resnet34 import ResNet34
import easydict
from keras.optimizers import Adam
from keras.utils import plot_model

# parameters

args = easydict.EasyDict({
    "height": 224,
    "width": 224,
    "ch" : 1,
    "K" : 4,
    "model_path": ""
})

lr = 0.01
epochs = 100

# you have to difine the data as *train_x, train_y, test_x, test_y* for training and testing


# build model
resnet = ResNet34(args).model

# save model architecture as image

# plot_model(resnet, show_shapes=True, to_file = 'resnet34.png')

## comfile model
resnet.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics = ['accuracy'])

# fit model
resnet_hist = resnet.fit(x=train_x,y=train_y,validation_split = 0.1, epochs = epochs)

## get performance
resnet_acc = resnet_hist.history['accuracy'][-1]
resnet_loss = resnet_hist.history['loss'][-1]

# test model
test_resnet_loss, test_resnet_acc  = resnet.evaluate(test_x, test_y)