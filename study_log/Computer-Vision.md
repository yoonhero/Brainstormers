# Computer Vision

## Index

- [Computer Vision](#computer-vision)
  - [Index](#index)
  - [CNN (Convolution Neural Network)](#cnn-convolution-neural-network)
    - [Convolution Layer 합성곱 층](#convolution-layer-합성곱-층)
    - [Pooling Layer 풀링 층](#pooling-layer-풀링-층)
    - [Flatten Layer](#flatten-layer)
    - [1D Convolution](#1d-convolution)
    - [2D Convolution](#2d-convolution)
    - [3D Convolution](#3d-convolution)
  - [VGGNet (2014)](#vggnet-2014)
  - [GoogLeNet (2014)](#googlenet-2014)
  - [ResNet: Deep Residual Learning for Image Recognition](#resnet-deep-residual-learning-for-image-recognition)
    - [Residual Block 잔여블록](#residual-block-잔여블록)
  - [Reference](#reference)

---

## CNN (Convolution Neural Network)

**What is convolution??**

컨볼루션, 한국 말로는 합성곱이라고 한다. 컨볼루션의 수학적인 의미는 신호를 커널을 이용해 국소적으로 증폭 또는 감소시켜서 정보를 추출 또는 필터링하는 것이다.

> z라는 필터(커널)을 움직여가면서 이미지나 신호에 컨볼루션 연산을 수행해서 정보를 추출할 수 있는 것이다. 

컨볼루션 연산은 매우 큰 행렬을 서로 곱하고 더하는 과정의 반복이기에 아주 오래걸려서 FFTConvolution 연산으로 빠르게 계산한다.

**CNN**

CNN은 Convolution Neural Network의 줄임말로 인간의 시신경을 모방하여 만든 딥러닝 구조 중 하나이다. 특히 convolution 연산을 이용하여 이미지의 공간적인 정보를 유지하고, Fully Connected Neural Network 대비 연산량을 획기적으로 줄였으며, 이미지 분류에서 좋은 성능을 보이는 것으로 알려져있다.

- 역사
    - 시신경의 구조
  
        1959년 시각 피질에 구조에 대한 연구에서 시각 피질 안의 많은 뉴런들이 작은 Local Receptive Field를 가진다는 사실을 밝혀냈다. Local의 의미는 보이는 것 중 일부 범위 안에 있는 시각 자극에만 반응한다는 의미이다. 여러 local receptive field의 영역은 서로 겹칠 수 있고, 이를 합치면 전체 시야를 감싸게 된다. 

    - Neocognition
  
        처음으로 convolution layer와 down sampling layer을 제안했다.

    - LeNet-5

        1998년 얀 르쿤 등이 수표에 쓰인 숫자를 인식하는 딥러닝 구조 LeNet-5를 발표하며 현대 CNN의 초석이 되었다.

**Image Data**

정형 데이터화라는 말은 컴퓨터가 식별가능한 형태로 데이터를 변환하는 것을 의미한다. 이미지는 픽셀 단위로 구성 되어 있고, 각 픽셀을 RGB 값으로 구성되어 있다. 즉 아주 작은 색이 담긴 네모 상자가 여러개가 모여 이미지가 되며, 색은 R, G, B의 강도의 합으로 표현할 수 있다. 이미지를 정형 데이터화 하는 방법은 컬러 이미지의 경우 **가로x세로x3** 의 배열로 나타낼 수 있으며 마지막 차원에서는 R, G, B 강도의 값을 나타낼 수 있다.


![im](https://e2eml.school/images/image_processing/three_d_array.png)


### Convolution Layer 합성곱 층

이미지의 feature가 될 수 있는 요소는 굉장히 많지만 그 중에서 대표적인 것이 형태가 될 것이다. 물체의 윤곽선만 보더라도 우리는 물체를 쉽게 가늠할 수 있다. 

![feature](https://media5.datahacker.rs/2018/10/features_3images.png)

Convolution Layer는 이러한 이미지의 특정 feature을 추출하기 위해서 등장하였다. 

더불어서 이미지의 특성상 각 픽셀 간에 밀접한 상관 관계를 가지고 있을 것이다. 이런 이미지 데이터를 flatten 해서 FC Layer 로 분석하게 된다면 데이터의 **공간적 구조** 를 무시할 수 있는 문제가 있다. 이미지 데이터의 공간적인 특성을 유지하기 위해서 Convolution Layer 가 등장하게 되었다.

Convolution Layer 는 sliding window 방식으로 가중치와 입력값을 곱한 값에 활성함수를 취하여 은닉층으로 넘겨준다.     

![convolutionlayer](https://user-images.githubusercontent.com/15958325/58780750-defb7480-8614-11e9-943c-4d44a9d1efc4.gif)

Convolution Layer는 전체를 인식하는 것이 아니라, 일부분을 투영하고 이를 우측의 수용영역에 연결되어 복합적으로 해석하는 것을 모방한 것이다. 이 Convolution Layer을 통해서 이미지 데이터를 flatten 해서 특성 1개 마다, 25개의 가중치를 학습해야 했지만, Convolution Layer을 도입하며 학습할 가중치가 9개로 줄은 것을 알 수 있다.


- 필터를 꼭 한 칸씩 이동해야 되는가?
- convolution을 반복적으로 수행하면 특성 배열의 크기가 점점 작아지지 않을까?
- 컬러 이미지는 어떻게 convolution을 할 수 있을까?

**Stride**

필터를 입력데이터나 특성에 적용할 때 움직이는 간격을 스트라이드라고 한다. 

![stride](https://i0.wp.com/developersbreach.com/wp-content/uploads/2020/08/Stride_item_banner.png?resize=760%2C505&ssl=1)

**Padding**

패딩은 반복적으로 합성곱 연산을 적용했을 때 특성의 행렬의 크기가 작아짐을 방지하는 것과 이미지의 모서리 부분의 정보손실을 줄이고자 이미지 주변에 0으로 채워 넣는 방법이다. 패딩처리를 하고 나면 이미지 모든 부분이 중심부처럼 합성곱 연산에 반영되어 정보 손실을 줄일 수 있다.

![padding](https://miro.medium.com/max/325/1*b77nZmPH15dE8g49BLW20A.png)

**컬러 이미지 합성곱**

Channel이 1인 이미지와 차이점은 필터의 채널이 3이라는 점과 RGB 각각에 다른 가중치로 Convolution을 하고 결과를 더해준다는 것이다. 




### Pooling Layer 풀링 층

풀링층은 데이터의 공간적인 특성을 유지하면서 크기를 줄여주는 층으로 연속적인 합성곱층 사이에 주기적으로 넣어준다. 데이터의 공간적인 특성을 그대로 유지하면서 크기를 줄여서 특정위치에서 큰 역할을 하는 특징을 추출하거나, 전체를 대변하는 특징을 추출할 수 있다는 장점이 있다. 

- Max Pooling 
  - Pooling Filter 영역에서 가장 큰 값을 추출하는 Pooling 방법이다.
- Average Pooling
  - Pooling Filter 영역에서 선택된 영역 값의 평균을 추출하는 Pooling 방법이다.

=> 대부분 Max Pooling 을 사용한다. (Max Pooling 을 이용한 결과가 더 좋다고 한다.)

### Flatten Layer

Flatten Layer 의 목적은 Convolution Layer, Pooling Lyaer를 feature를 추출한 다음에는 추출된 특성을 Output Layer에 연결하여 어떤 이미지인지 분류하기 위함이다. 즉, flattening layer 이후로는 일반 신경망 모형과 동일하다. 



**1D Convolution vs 2D Convolution vs 3D Convolution**

합성곱은 이동하는 방향의 수에 따라서 1D, 2D, 3D로 분류할 수 있다. 

### 1D Convolution

![1D Convolution](https://github.com/ifding/learning-notes/raw/master/machine-learning/images/1d-convolutions.png)

1D Convolution은 시계열 데이터 예를 들면, Biomedical Signals (EEG 또는 ECG)나 금융데이터, Biometrics (e.g. voice, signature and gesture), video processing, music mining, forecasting weather 와 같은 분야에 사용할 수 있다고 한다. 

- 오로지 1 방향으로 합성곱 연산을 진행한다.
- example, input = [1, 1, 1, 1, 1] filter = [0.25, 0.25, 0.25] output = [1, 1, 1, 1, 1]
- output-shape 도 1차원 배열이 된다.

**활용예시**

UCI에서 진행한 사람의 움직임을 자이로 센서로 9 채널로 입력을 받아서 1D Convolution으로 예측을 진행한다.

> (batch, seq_len, n_channels)

- seq_len: 시계열 단계의 길이
- n_channels: 관찰한 채널 수

1D Convolution을 활용하여 텍스트 분류 [more information](https://arxiv.org/pdf/1510.03820.pdf)


### 2D Convolution

![2d Convolution](https://github.com/ifding/learning-notes/raw/master/machine-learning/images/2d-convolutions.png)

- 2 방향으로 합성곱 연산을 진행한다.
- output-shape 은 2차원 행렬이 된다.
- example) computer vision, edge detection algorithm, Sobel Edge Filter


2D Convolutions with 3D input - LeNet VGG ...

- H*W*C로 3차원 데이터이다.
- output-shape 은 2차원 행렬이다.
- 그래서 필터의 depth = L 이어야 한다.
- 두 방향으로만 합성곱 연산을 진행한다. 
- 3차원 데이터 -> 2차원 * N 행렬


### 3D Convolution

![3D Convolution](https://github.com/ifding/learning-notes/raw/master/machine-learning/images/3d-convolutions.png)

- 3방향으로 합성곱 연산을 진행한다.
- output-shape는 3차원이다.
- d < L
- example) C3D video descriptor 


## VGGNet (2014)

깊이가 깊어지면 overfitting, gradient vanishment, 연산량 문제가 생기기 때문에 깊이를 증가시키는 것이 쉬운 문제는 아니었다. VGGNet의 핵심 내용은 다음과 같다.

- 깊이를 증가시키면 정확도가 좋아진다.
- 3x3 필터를 여러 겹 사용하여 크기가 큰 필터를 분해하면 추가적인 비선형성을 부여하고 parameter 수를 감소시킨다.
- pre-initialization을 이용하면 모델이 빠르게 수렴한다.
- data augmentation(resize, crop, flip)을 적용하면 다양한 scale로 feature을 포착할 수 있다.

VGG 연구자들은 3x3 필터 사이즈만을 사용하면서 엄청난 사실을 발견했다. 3x3 convolutional filter를 2개 이용하면 5x5 convolutional, 3개 이용하면 7x7 convolutional이 된다. 3x3 filter를 여러 겹 이용하게 되면 하나의 relu 대신 2개, 3개의 relu를 이용할 수 있고, parameter 수를 감소시킬 수 있다고 한다. 

1x1 conv layer을 사용한 이유는 비선형성을 부여하기 위해서라고 한다. 입력과 출력의 channels을 동일하게 하고 1x1 conv layer를 이용하면 relu 함수를 거치게 되어 추가적인 비선형성이 부여된다.

## GoogLeNet (2014)

GoogLeNet은 sparse한 data 구조에 집중한다. NIN이라는 논문에서는 높은 상관관계에 있는 뉴런들을 군집화 시키고 마지막 계층에서 활성화 함수들의 상관관계를 분석함으로써 최적의 network topology를 구축할 수 있었다고 말한다. NIN을 간단히 말하면 convolution을 수행할 때, 수행 후 feature map을 얻게 되는데, multilayer perceptron 네트워크를 convolution시 추가로 적용하여 feature map을 만든다. 

이를 통해 fully connected layer와 convolution layer를 dense 구조에서 sparse 구조로 바꿀 수 있었다고 말한다. GoogLeNet에서는 NIN을 구현하기 위해 Inception module을 적용하였다고 한다.

**Inception Module**

Inception module의 주요 아이디어는 convolutional network에서 sparse 구조를 손쉽게 dense 요소들로 근사화하고 다룰 수 있는 방법을 찾는 것에서 근거한다. 

Inception module에서 feature map을 효과적으로 추출하기 위해 1x1, 3x3, 5x5의 convolution 연산을 각각 수행하며, 각각 Matrix의 height, width가 같아야 하므로 pooling 연산에서 padding을 추가해준다. 

**Auxiliary Classifier**

네트워크의 깊이가 깊어지면 깊어질수록 vanishing gradient 문제를 피하기 어려워지는데 이 문제를 극복하기 위해서 네트워크 중간에 보조 분류기(Auxiliary Classifier)을 달아주었다.


## ResNet: Deep Residual Learning for Image Recognition

> 신경망의 깊이가 엄청나게 깊어질 수 있는 간단한 아이디어를 제시!

**TLDR;**

- 깊은 네트워크를 학습시키기 위해서 *Residual Learning*을 제시! 

![previous](https://velog.velcdn.com/images%2Fgood159897%2Fpost%2Fd2ff36db-1f01-4de9-8c07-eb89283785aa%2FResNet%20%EC%9D%B4%EC%A0%84_2.PNG)

VGG network와 Googlenet이 나오면서 신경망이 깊으면 깊을수록 성능이 올라간다는 것은 우리 눈의 시신경 세포가 100만개 정도 되는 것처럼 더 다양한 특성을 추출하고 처리할 수 있기 때문이다. 하지만 전에 제시된 방법으로는 신경망의 깊이가 깊어지면 오히려 성능이 저하되는 역효과가 일어났다. 더 깊은 신경망을 쌓을 수 있는 매우 이해하기 쉽지만 아주 중요한 아이디어를 이 논문에서 제시했다고 볼 수 있다. 

### Residual Block 잔여블록

![resnet](https://thebook.io/img/080263/233_2.jpg)

Residual Block의 역할은 바로 Optimization의 난이도를 낮추는 것이다. 

위 그림의 왼쪽 부분은 기존 방식이고 오른쪽 부분이 Resnet 논문에서 제시된 방법이다. 기존 방식에서는 $x$를 $H(x)$를 통해서 바로 특징을 추출하는 것을 학습했다면 새로운 방식에서는 $x$를 합성곱층을 표현한 $F(x)$와 더하여 학습해야하는 $H(x)$의 Optimization의 난이도를 줄일 수 있었다. 쉽게 말하면 바로 아무것도 없는 상태에서 무언가를 하기는 어려울 수 있지만 우리가 전에 쌓은 지식을 주면서 학습해 나간다면 학습이 더 수월할 수 있다는 것이다. 결론적으로 $x$ identity를 더해서 앞서 학습된 정보를 포함하면서 $F(x)$만 학습하면 Optimization의 성능이 더 좋고 수렴 속도가 빨랐다고 한다. 


## Reference

- CNN

    - [https://ydy8989.github.io/2021-02-02-conv/](https://ydy8989.github.io/2021-02-02-conv/)
    - [https://www.youtube.com/watch?v=KuXjwB4LzSA&t=11s](https://www.youtube.com/watch?v=KuXjwB4LzSA&t=11s)
    - [https://89douner.tistory.com/57](https://89douner.tistory.com/57)
    - [https://yjjo.tistory.com/8](https://yjjo.tistory.com/8)
    - [https://github.com/ifding/learning-notes/blob/master/machine-learning/1d-2d-and-3d-convolutions-in-cnn.md](https://github.com/ifding/learning-notes/blob/master/machine-learning/1d-2d-and-3d-convolutions-in-cnn.md)
    - [https://statinknu.tistory.com/26](https://statinknu.tistory.com/26)
    - [https://deep-learning-study.tistory.com/389?category=963091](https://deep-learning-study.tistory.com/389?category=963091)
    - [https://deep-learning-study.tistory.com/523](https://deep-learning-study.tistory.com/523)
