# LSTM

## 1.LSTM의 배경

LSTM은 RNN에서 파생된 것으로, 기존 RNN의 장기 의존성(long-term dependency)문제를 해결하기 위해 등장하였다.

- _장기 의존성이란? (long-Term Dependency)_
  _ -nn은 타임스텝 t에서 이전 타임스텝인 t-1의 상태를 입력으로 받는구 조다. 그러면 이전의 정보가 현재의 타임스텝 t에 영향을 주게된다.하 지 타임스텝이 길어질수록 입력 데이터가 rnn cell을 거칠 때마다 연산을통 해데이터가 변환되어 일부 정보가 소실되어 이후 타임스텝에 영향을 주지못 하 문제가 발생하게 되는데, 이를 장기 의존성 문제 (long-termd eendency) 라고 한다._

## 2. RNN이란?(Recurrent Neural Network)

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2fname=http%3A%2F%2Fcfile26.uf.tistorycom%2Fimage%2F99B366365ACB86A014D902">

<img src="http://i.imgur.com/s8nYcww.png"/>

RNN의 기본 모형.
RNN은 X\_{t}가 입력되어 A라는 신경망을 거쳐 Yt라는 출력값을 내놓는다.
위의 첫 이미지를 보았을 때, 입력데이터가 한덩어리의 신경망을 거쳐 출력이되고 현재의 상태가 다음 과정에 전달되는 일련의 과정이 반복된다. 이는RNN이 이전 데이터를 기억해 이후 데이터에 영향을 줄 수 있는 형태임을의미한다.
두번째 이미지에서
녹색박스 = 히든 state
빨간 박스 = 인풋 x
파란 박스 = 아웃풋 y
현재 상태의 히든 state h <sub>t</sub> 는 직전 시점의 히든 state ht-1을받아 갱신된다.
현재 상태의 아웃풋 y <sub>t</sub> 는 h <sub>t</sub> 를 전달받아갱신된다.
히든 state 활성함수(activation function) : 비선형 함수인 하이퍼볼릭탄젠트(tanh)

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2fname=http%3A%2F%2Fcfile3.uf.tistorycom%2Fimage%2F9901A1415ACB86A0211095"/>
기본 모형을 시간순으로 풀어본 모형.

위의 이미지처럼, 반복을 풀어봤을 때 좀 더 이해하기 쉬운데 RNN의 체인처럼이어지는 성질덕분에 음성이나 문자 등 순차적으로 등장하는 데이터 처리에적합하다고 알려져 있다. 또한 시퀀스 길이에 관계없이 인풋과 아웃풋을받아들일 수 있는 네트워크 구조라 필요에 따라 다양하고 유연하게 구조를만들 수 있다는 장점이 있다.

## 3. LSTM

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile5.uf.tistory.com%2Fimage%2F99893B375ACB86A035DE41"/>
RNN의 반복모듈이 단 하나의 layer를 갖고 있는 일반적인 모습이다.

LSTM도 RNN과 같이 체인과 같은 구조를 가지고 있지만, 각 반복 모듈은 다른 구조를 갖고있다. 4개의 layer가 서로 정보를 주고 받는다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile30.uf.tistory.com%2Fimage%2F999F603E5ACB86A00550F0"/>

_기호에 대한 정리_

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile5.uf.tistory.com%2Fimage%2F993A93495ACB86A02FFAA8">

- _선 : 한 노드의 아웃풋을 다른 노드의 인풋으로 벡터 전체를 보내는 흐름._
- _분홍색 동그라미 : 벡터 합과 같은 pointwise operation_
- _노란색 박스 : 학습된 neural network layer_
- _합쳐지는 선 : concatenation_
- _갈라지는 선 : fork_

### LSTM의 동작 단계

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile2.uf.tistory.com%2Fimage%2F9957DB445ACB86A02155EA" />

_lstm의 forget gate layer_

1. 첫번째로 cell state로부터 어떤 정보를 버릴 것인지 정한다. 무엇을버릴지는 "**forget gate layer** " 라고 불리는 sigmoid layer에 의해결정된다. 이 단계에서 h <sub>t-1</sub>, x <sub>t</sub> 를 받아서 0 과1사이의 값을 C <sub>t-1</su b>에 보낸다. ( 1이면 모든정보보존, 0이면모든정보버림.)

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile4.uf.tistory.com%2Fimage%2F99D969495ACB86A00BFC15">

_lstm의 input gate layer_

2. 앞으로 들어오는 새로운 정보 중 어떤 것을 cell state에 저장할 것인지정한다. "**input gate layer**" 라고 불리는 sigmoid layer가 어떤 값을업데이트할지 정하고 tanh layer가 새로운 후보값들의 vector를 만든다.이렇게 두 단계에서 나오는 정보를 합쳐 state 를 갱신할 준비를 한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile9.uf.tistory.com%2Fimage%2F997589405ACB86A00CADEA">

3. 이제 과거 state 인 C<sub>t</sub> 를 업데이트 해 새로운 cell state인C <sub>t</sub> 를 만든다.  
   먼저, 이전 state에 f<sub>t</sub> 를 곱해서 버리고자 했던 정보를 버리고난 후 i<sub>t</sub>\*C~<sub>t</sub>를 더한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile9.uf.tistory.com%2Fimage%2F997589405ACB86A00CADEA">
4. 마지막으로 무엇을 output으로 내보낼 지 정한다. 
sigmoid layer에 input 데이터를 넣어서 state의 어느 부분을 output으로 내보낼지 정한다. 그리고 cell state를 tanh에 넣어 -1rhk 1사이 값을 받은 뒤 계산한 sigmoid gate의 output과 곱해준다. 그러면 우리가 내보내고자하는 cell state를 바탕으로 필터된 값을 ouput으로 보낼 수 있다.

### LSTM class

```
tf.keras.layers.LSTM(
    units,
    activation="tanh",
    recurrent_activation="sigmoid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    implementation=2,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    time_major=False,
    unroll=False,
    **kwargs
)
```

### example code

라이브러리

```
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

```

단층 LSTM

```
from keras.layers import LSTM

def lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape = (49,1), return_sequences = False))
    model.add(Dense(46))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model

model = KerasClassifier(build_fn = lstm, epochs = 200, batch_size = 50, verbose = 1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_test_ = np.argmax(y_test, axis = 1)
print(accuracy_score(y_pred, y_test_))


```

다층 LSTM

```
def stacked_lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape = (49,1), return_sequences = True))
    model.add(LSTM(50, return_sequences = False))
    model.add(Dense(46))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model

model = KerasClassifier(build_fn = stacked_lstm, epochs = 200, batch_size = 50, verbose = 1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_test_ = np.argmax(y_test, axis = 1)
print(accuracy_score(y_pred, y_test_))

```
