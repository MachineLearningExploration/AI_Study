import math
import os

import numpy as np
from keras import Input, initializers, layers, metrics, models, optimizers, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    MaxPooling2D,
    ZeroPadding2D,
)
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

import keras.applications.resnet_v2 as r

class ResNet101:
    def __init__(self, args):
        self.K = args.K
        self.width = args.width
        self.height = args.height
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
        x = Activation("relu")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)

        return x

    def conv2_layer(self, x):
        x = MaxPooling2D((3, 3), 2)(x)

        shortcut = x

        for i in range(3):
            if i == 0:
                shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding="valid")(
                    shortcut
                )
                shortcut = BatchNormalization()(shortcut)

            x = Conv2D(64, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(256, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation("relu")(x)

            shortcut = x

        return x

    def conv3_layer(self, x):
        shortcut = x

        for i in range(4):
            if i == 0:
                x = Conv2D(128, (1, 1), strides=(2, 2), padding="valid")(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)

                shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding="valid")(
                    shortcut
                )
                shortcut = BatchNormalization()(shortcut)

            else:
                x = Conv2D(128, (1, 1), strides=(1, 1), padding="valid")(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(512, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation("relu")(x)

            shortcut = x

        return x

    def conv4_layer(self, x):
        shortcut = x

        for i in range(23):
            if i == 0:
                shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding="valid")(
                    shortcut
                )  # increase the dimension
                shortcut = BatchNormalization()(shortcut)

                x = Conv2D(256, (1, 1), strides=(2, 2), padding="valid")(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)

            else:
                x = Conv2D(256, (1, 1), strides=(1, 1), padding="valid")(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)

            x = Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation("relu")(x)

            shortcut = x

        return x

    def conv5_layer(self, x):
        shortcut = x

        for i in range(3):
            if i == 0:
                x = Conv2D(512, (1, 1), strides=(2, 2), padding="valid")(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)

                shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding="valid")(
                    shortcut
                )
                shortcut = BatchNormalization()(shortcut)

            else:
                x = Conv2D(512, (1, 1), strides=(1, 1), padding="valid")(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation("relu")(x)

            shortcut = x

        return x

    def build_model(self):
        input_tensor = Input(
            shape=(self.width, self.height, self.ch), dtype="float32", name="input"
        )

        x = self.conv1_layer(input_tensor)
        x = self.conv2_layer(x)
        x = self.conv3_layer(x)
        x = self.conv4_layer(x)
        x = self.conv5_layer(x)

        x = GlobalAveragePooling2D()(x)
        output_tensor = Dense(self.K, activation="softmax")(x)

        resnet = Model(input_tensor, output_tensor)
        resnet.summary()

        return resnet


if __name__ == "__main__":
    resnet = ResNet101()

    # plot_model(resnet, to_file == "resnet101.png", show_shapes=True)
