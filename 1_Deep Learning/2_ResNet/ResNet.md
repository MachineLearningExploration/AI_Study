####  Reference

*  https://medium.com/@codecompose/resnet-e3097d2cfe42
*  https://datascienceschool.net/view-notebook/958022040c544257aa7ba88643d6c032/
*  https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33

This is written by Hyemee Kim(khmee0219@google.com)

#### Intro

최근 CNN의 기술을 통한 분류는 사람이 가능한 수준 혹은 그 이상으로 발전하였다. 이를 통해 발전한 네트워크 구조는 이미지를 중심으로 발전하였다. 하지만, 이제는 이미지 이외의 다양한 성격의 데이터에 활용될 수 있는 것을 알고, 다양한 시도가 이루어지고 있다. 이를 통해 서로 다른 성격의 딥러닝 모델들이 빠른 속도로 발전하고 있으며, 높은 성능으로 다양한 곳에서 활용되고 있다.

그 중 ResNet은 layer가 많아질 수록 손실되거나 확장되는 문제를 해결하기 위해 사용되는 모델인 만큼, 다른 모델들에서도 충분히 활용가능한 컨셉으로 생각하여 스터디를 진행하였다.

ResNet 이전의 이미지 처리 Deep Learning 알고리즘은 다음과 같다.

(Reference: https://medium.com/analytics-vidhya/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5)

![History](https://miro.medium.com/max/1050/1*DBXf6dzNB78QPHGDofHA4Q.png)

* LeNet-5(1998)
* AlexNet(2012)
    * 해당 알고리즘은 병렬로 구성된 CNN을 기반으로 한다. 이는 학습속도를 빠르게 하기 위함이나, 장비의 부족으로 시도해보지 못했다.

* ZFNet(2013)
* GoogLeNet/Inception(2014)
    * 발표 이후 높은 성능을 보였으나, 무거운 모델 구조로 ResNet보다 선호도가 낮다.

* VGGNet(2014)
* ResNet(2015)


#### ResNet의 특징

기존 Convolution Deep Learning Network는 네트워크가 깊어짐에 따라 gradient값이 소실(vanishing)되거나 확장(exploding)되는 문제가 발생한다. 하지만, Deep learning 모델은 실험을 통해 layer가 깊어질 수록 좋은 성능을 보이는 것이 밝혀져 있다. 즉, 좋은 성능의 모델을 만들기 위해서는 layer를 많을 수록 좋으나, 이는 모델 성능을 저하시키는 원인이 된다.

더 깊은(=layer가 많은) 모델을 gradient 문제 없이 활용하기 위해 사용되기 시작한 방법이 ResNet(=Residual Network)이다. layer를 통해 계산된 값인 F(x)에 입력값인 x를 다시 더한 값을 활용하여 학습을 수행한다.

#### 모델 구조

* skip/shortcut 구조

![shortcut](https://miro.medium.com/max/1050/1*G8e3wym0Rs1yPcp62yBgaQ.png)

ResNet은 Residual을 활용한 네트워크 모델이라하여 붙여진 이름이다. 일반적으로 Residual을 실제값과 예측값의 차이를 의미한다. 여기서 Residual은 F(x)를 의미하며, shortcut을 통해 도출된 값을 H(x)라 하면, 다음과 같은 수식이 성립한다.

$$ H(x) = F(x) + x $$
$$ F(x) = H(x) - x $$

즉, Residual인 F(x)를 학습하는 네트워크 모델이라하여 ResNet이라는 이름을 갖게 된 것이다.

이와 같은 과정을 통해 Gradient가 소실 및 확장되더라도, 입력값 x를 더하는 과정을 통해 이 문제를 해결할 수 있다. 이 때, 입력값 x는 identity x라 한다.

* Bottleneck 디자인

![Bottleneck](https://miro.medium.com/max/1500/1*f7C6lhx50ol9oYifAOu5vw.png)

ResNet는 굉장히 깊기 때문에 복잡도가 높은 모델이다. 이와 같은 복잡도를 낮추기 위해 bottleneck 디자인을 Network In Network와 GoogLeNet(Inception-v1)에서 차용되었다. 이 디자인은 네트워크의 첫번째 및 마지막 레이어에 1x1 conv레이어를 추가한 형태이다. 이와 같은 구조는 네트워크 성능을 유지하며 parameter의 수를 줄일 수 있는 것으로 알려져 있다.

이 같은 Bottleneck 디자인은 복잡도가 높은 ResNet50 모델부터 더 큰 사이즈를 갖는 ResNet101, ResNet152에 활용된다.

#### ResNet의 종류

일반적으로 알려져있는 ResNet은 총 5가지가 있다. 각각 18개, 34개, 50개, 101개, 152개의 layer를 가지며, 50개, 101개, 152개의 layer를 갖는 ResNet만 Bottleneck 디자인을 활용하였다.

![ResNet Type](https://miro.medium.com/max/1500/1*ijr3YZG5oyvz3Mr52k-RQg.png)

Python 및 Keras를 활용하여 만든 ResNet은 각각 다음과 같다.
(각 모델의 구조는 여기에 올리기에 너무 길어 링크로 대체한다.)

* [ResNet18](./resnet18.py) ~ [구조](./resnet18.png)
* [ResNet34](./resnet34.py) ~ [구조](./resnet34.png)
* [ResNet50](./resnet50.py) ~ [구조](./resnet50.png)
* [ResNet101](./resnet101.py) ~ [구조](./resnet101.png)
* [ResNet152](./resnet152.py) ~ [구조](./resnet152.png)

(해당 코드는 [Run](./Run.py)를 활용하여 실행할 수 있다.)

#### Ongoing

- 이미지 분류
    
```
활용한 이미지 데이터는 일관된 특성을 관찰하기가 매우 어려워 분류의 성능이 매우 떨어졌다. 기존에 전통적인 CNN 모델에 다양한 filter를 사용하여 분류를 시도해보고, Autoencoder를 활용하여 특징을 추출하는 전처리 과정을 거쳐 이미지 분류를 시도해보았으나, 모두 결과가 좋지 않았다.

이미지 crop 사이즈를 줄여 ResNet 기반의 분류 작업을 수행하고 있다. 전통적인 CNN 방법론과 성능을 비교하였을 때, 학습속도가 월등히 빠르며, 높은 성능을 보여주고 있다. 현재로서는 overfitting하는 결과를 보이며, 이는 학습데이터가 부족하여 발생하는 것으로 추정된다. 추가적인 데이터를 확보하여 학습을 진행해볼 예정에 있다. 
```