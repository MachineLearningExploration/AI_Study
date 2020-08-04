# CNN

Convolutional Neural Network

![https://cs231n.github.io/assets/cnn/convnet.jpeg](https://cs231n.github.io/assets/cnn/convnet.jpeg)

## Convolution

![http://ufldl.stanford.edu/tutorial/images/Convolution_schematic.gif](http://ufldl.stanford.edu/tutorial/images/Convolution_schematic.gif)

Convolution(합성곱)은 인접한 데이터를 섞어서 새로운 값을 구해나는 연산이다.

## Filter (Kernel)

![http://en.wikipedia.org_wiki_Kernel_(image_processing).png](./en.wikipedia.org_wiki_Kernel_(image_processing).png)

적절한 필터를 사용해 데이터의 특징을 추출해낼 수 있으며, 다양한 필터를 중첩해서 Convolutional 레이어를 구성한다. CNN에서는 학습을 통해 자동으로 적합한 필터를 적용한다.

## Pooling

![http://ufldl.stanford.edu/tutorial/images/Pooling_schematic.gif](http://ufldl.stanford.edu/tutorial/images/Pooling_schematic.gif)

![https://cs231n.github.io/assets/cnn/pool.jpeg](https://cs231n.github.io/assets/cnn/pool.jpeg)

![https://cs231n.github.io/assets/cnn/maxpool.jpeg](https://cs231n.github.io/assets/cnn/maxpool.jpeg)

Pooling 레이어는 Convolutional 레이어의 출력을 입력 데이터로 사용해 특정 데이터를 강조하는 역할을 하는 동시에 연산할 이미지의 크기가 줄어드는 이점을 얻을 수 있다. CNN에서는 최대 값을 뽑아내는 Max Pooling을 주로 사용한다.

## Example

```python
import keras
from keras import layers
from keras.utils import plot_model

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
```

## References

- <https://cs231n.github.io/convolutional-networks/>
- <https://en.wikipedia.org/wiki/Kernel_(image_processing)>
- <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>
- <http://ufldl.stanford.edu/tutorial/supervised/Pooling/>
