# Multi Layer Perceptron

## 회귀모델(Regression)

$$Y = XW + b$$

우선적으로 입력값을 활용하여 특정 값을 예측하는 문제는 위와 같이 수식화할 수 있다. 위 수식에서 X는 데이터, Y는 예측값을 의미한다. 

MLP 역시 간단히 표현하면 다음과 같이 위의 수식과 유사하게 표현될 수 있다.

$$Y = f(X) + b$$





## MLP

![mlp](https://miro.medium.com/max/1050/1*-IPQlOd46dlsutIbUq1Zcw.png)

위 그림은 단순하게 만들어진 MLP 모형이다. 이 때 Hidden layer를 더 많이 쌓으면 더 복잡한 수식을 표현할 수 있다.

즉, 레이어의 수와 모델의 복잡도는 비례하게 되며, 레이어의 수에 따라 문제가 발생할 수 있는 서로 다르다. 레이어의 수가 적은 경우 모델이 단순하여 충분히 모델이 학습되지 않는 underfitting 문제가 발생할 수 있다. 반면, 레이어의 수가 많은 경우 모델이 너무 복잡하여 학습 데이터에 대해 과도하게 학습되고 다른 데이터에 대한 예측 성능이 떨어지는 overfitting 문제가 발생할 수 있다.

underfitting의 경우 학습 횟수인 epoch나, layer 수를 늘려 해소할 수 있으며, overfitting 문제는 dropout을 통해 해결할 수 있다. Dropout은 각각의 텐서가 연결되어 있는 링크의 일부를 제거하는 방법으로, fully-conntected 모델보다 복잡한 수준을 낮추어 overfitting 문제를 해결하게 된다.

#### (참고) 단층 레이어 퍼셉트론

- 입출력 레이어만 존재하는 모델
- 이는 비선형 모델을 표현하기 어려움

### Hyper Paramters of MLP

- the number of hidden layers' units
- the number of hidden layers
- Optimizer
- loss function
- activation funtions of layers


## Example code

### Keras

```
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()

model.add(Dense(hidden[0], activation='relu', input_shape=(model_in,), name='Hidden-1'))
model.add(Dense(hidden[1], activation='relu', name='Hidden-2'))
model.add(Dense(hidden[2], activation='relu', name='Hidden-3'))
model.add(Dense(model_out, activation = 'softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy')
```

### Pytorch

```
추가 예정
```