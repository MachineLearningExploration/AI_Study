from keras import models, layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, \
    ZeroPadding2D, Add

import os
import matplotlib.pyplot as plt
import numpy as np
import math


class ResNet50:
    def __init__(self, args):
        self.K = args.K
        self.height = args.height
        self.width = args.width
        self.ch = args.ch

        if os.path.exists(args.model_path):
            self.model = models.load_model(filepath=args.model_path)
            print(f"Loaded model {args.model_path}")
        else:
            self.model = self.build_model()

    def conv1_layer(self, x):
        x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(64, (7, 7), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)

        return x

    def conv2_layer(self, x):
        x = MaxPooling2D((3, 3), 2)(x)

        shortcut = x

        for i in range(3):
            if (i == 0):
                shortcut = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(shortcut)
                shortcut = BatchNormalization()(shortcut)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        return x

    def conv3_layer(self, x):
        shortcut = x

        for i in range(4):
            if (i == 0):
                x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                shortcut = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                shortcut = BatchNormalization()(shortcut)

            else:
                x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        return x

    def conv4_layer(self, x):
        shortcut = x

        for i in range(6):
            if (i == 0):
                shortcut = Conv2D(256, (3, 3), strides=(2, 2), padding='valid')(shortcut)  # increase the dimension
                shortcut = BatchNormalization()(shortcut)

                x = Conv2D(256, (3, 3), strides=(2, 2), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

            else:
                x = Conv2D(256, (3, 3), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

            x = Conv2D(256, (3, 3), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        return x

    def conv5_layer(self, x):
        shortcut = x

        for i in range(3):
            if (i == 0):
                x = Conv2D(512, (3, 3), strides=(2, 2), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                shortcut = Conv2D(512, (3, 3), strides=(2, 2), padding='valid')(shortcut)
                shortcut = BatchNormalization()(shortcut)


            else:
                x = Conv2D(512, (3, 3), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        return x

    def build_model(self):
        input_tensor = Input(shape=(self.height, self.width, self.ch), dtype='float32', name='input')

        x = self.conv1_layer(input_tensor)
        x = self.conv2_layer(x)
        x = self.conv3_layer(x)
        x = self.conv4_layer(x)
        x = self.conv5_layer(x)

        x = GlobalAveragePooling2D()(x)
        output_tensor = Dense(self.K, activation='softmax')(x)

        resnet34 = Model(input_tensor, output_tensor)
        # resnet50.summary()

        return resnet34