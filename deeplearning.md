## Embedding Layer

![embedding](./docs/torch-embedding.jpeg)
 
우리가 사용하는 언어나 이미지는 0과 1로만 이루어진 컴퓨터 입장에서 그 의미를 파악하기가 어렵다. 예를 들어 우리가 인공지능 챗봇을 제작한다고 하자. 우리가 입력한 말을 과연 컴퓨터가 바로 이해할 수 있을까? 우리는 우리가 입력한 말을 tokenize하는 과정을 통해서 '언어의 벡터화'를 한다. 이런 tokenize하는 일련의 과정을 **Word Embedding**이라고 한다.

가장 흔하게 사용되는 벡터화 방법은 One-hot Encoding 이다.

- One-hot Encoding
	- 필요한 정보를 담은 하나의 값만 1로 두고, 나머지 값은 0으로 설정한다. 
	- 대부분의 값이 0을 갖고 단 한 개의 1인 값을  가지는 일종의 Sparse Matrix (희소 행렬)로 표현된다.
	- 단어가 많아질수록 벡터 공간만 커지는 비효율적인 방법이다.
	- 원-핫 인코딩은 단어가 무엇을 의미하는지 설명하지 못한다.

-> 이런 문제가 있었기에 Dense Matrix로 변환하는 표현법이 제시되었다.

사람이 사용하는 언아나 이미지를 컴퓨터에게 이해시키기 위해서는 어떤 **벡터 공간**에 우리가 표현하고자 하는 정보를 mapping해야 한다.

고차원의 정보를 저차원으로 변환하면서 필요한 정보를 보존하는 것을 **임베딩(Embedding)** 이라고 한다. 

### 텍스트 임베딩

- 임베딩 벡터
	- 밀집행렬로 임베딩된 벡터는 각 요소에서 단어의 서로 다른 특성을 나타낸다.
	- 각 요소에는 단어가 관련 특성을 대표하는 정도를 나타내는 0~1 사이의 값이 포함된다.
	- 이런 임베딩을 통해 텍스트를 단순히 '구분' 하는 것이 아닌 의미적으로 '정의'하는 것이라고 볼 수 있다.
	 ![text-embedding](https://velog.velcdn.com/images%2Fdongho5041%2Fpost%2F6cff5fbf-2a1c-42c1-8a08-613c4582729d%2Fimage.png)


### 이미지 임베딩

이미지의 경우 텍스트와 달리 이미지 데이터가 그 자체로 밀집행렬이라고 볼 수 있다.

하지만 이런 고차원 고밀도의 데이터를 일일이 비교해가며 비슷한 이미지를 찾는다는 것은 매우 비효율적이다. 그러므로 이미지의 저차원적 특성 벡터를 추출해 이미지에 포함된 내용이 무엇인지 나타내는 일정한 지표를 얻어 효과적으로 비교한다.


**Pytorch 코드**

```python

torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type)

```

- num_embeddings: 임베딩 벡터를 생성할 전체 범주의 개수

- embedding_dim: 임베딩 벡터의 차원

- padding_idx: 지정된 인덱스에 대해서는 학습이 진행되지 않는다.

- max_norm: 특정 실수가 주어지고 임베딩 벡터의 norm이 이 값보다 크면 norm이 이 값에 맞추어지도록 정규화된다.

  
## Regularization

<strong>Batch</strong>는 한 에포크에 모든 데이터를 신경망이 처리하고 평균 기울기로 경사하강법을 진행하면 메모리가 부족할 수 있어서 데이터셋을 나눠서 학습을 진행하는 단위를 말한다.

학습 데이터를 Batch로 나누어서 Stochastic Gradient Descent 를 실시한다. => 효율적인 학습이 가능하다.

-   학습데이터 전체 학습을 끝내면 한 Epoch가 끝난다.
-   Gradient를 구하는 단위는 Batch.

### Problem

Batch로 나누어서 학습을 진행할 경우 Internal Covariant Shift 문제나 fully connected 연산 이후에 데이터의 분포가 달라질 수 있다는 문제를 가지고 있다. Internal Covariant Shift 문제는 학습 과정에서 계층 별로 입력 데이터의 분포가 달라지는 현상이다. 또 각 계층에서 입력 받은 feature은 convolution 연산이나 fully connected 연산을 거치고 데이터 분포가 달라질 수 있다.

![internal_covariant_shift](https://miro.medium.com/max/678/1*BSssXFdw2MWR3SqdGF-BoQ.png)

<strong>내부 공변량 변화(Internal convariate Shift)</strong>

배치 정규화를 제안한 논문에서는 기울기 소실/폭주 등의 딥러닝 모델의 불안정성이 층마다 입력의 분포가 달라지기 때문이라고 주장했다. 

- 공변량 변화는 훈련데이터의 분포와 테스트 데이터의 분포가 다른 경우를 의미
- 내부 공변량 변화는 신경망 층 사이에서 발생하는 입력 데이터의 분포 변화를 의미한다.

### Solution

-   Change Activation Function
-   Careful Initialization - He initialization, Xavier Initialization
-   Small Learning rate
-   Batch Normalization

### Change Activation Function

활성화 함수가 필요한 이유는 바로 선형적으로 연결된 신경망에 활성화 함수가 없다면 아무리 신경망이 깊어지더라도 선형적인 관계만 나타낼 수 있기 때문이다. 비선형적인 데이터의 분류등을 진행하기 위해서 활성화 함수가 각 계층마다 필요한 것이다.


### Careful Initialization

신경망 학습에서 특히 중요한 것은 가중치의 초깃값이다. 가중치의 초깃값을 무엇으로 설정하느냐가 신경망 학습에 영향을 많이 끼친다. 그 이유는 각 뉴런의 가중치를 기반으로 에러를 결정하기 때문이고, 정확한 모델을 얻으려면 작은 에러를 필요로 하기 때문이다. 


<strong>Weight Decay</strong>

가중치 감소(Weight Decay) 기법은 overfitting을 억제해 성능을 높이는 기술이다. 

Weight Decay는 loss function에 L2 Norm과 같은 penalty를 추가하는 정규화 기법이다. 

- Norm
    Norm은 벡터의 크기를 측정하는 방법이다. 두 벡터 사이의 거리를 측정하는 방법이기도 하다. 
    
    ![norm](https://wikimedia.org/api/rest_v1/media/math/render/svg/811da8c9721a21d9c3e638e2c30884adc9c38c5b)

    - L1 Norm
        L1 Norm 은 벡터 p, q의 각 원소들의 차이의 절대값의 합이다.
        L1 Regularization 은 cost function에 가중치의 절대값을 더해준 것을 cost로 사용하는 것으로 가중치의 크기가 너무 크지 않는 방향으로 학습 되도록 한다. L1 Regularization을 사용하는 Regression model을 Least Absolute Shrinkage and Selection Operator Regression 이라고 부른다. 

    - L2 Norm
        L2 Norm은 벡터 p, q의 유클리디안 거리이다. 
        L2 Regularization 도 L1 Regularization 과 비슷하게 가중치가 너무 크지 않는 방향으로 학습되게 된다. 이를 weight decay라고도 한다.

    L2 Norm은 직관적으로 오차의 제곱을 더해서 L1 Norm보다 Outlier에 더 큰 영향을 받는다. 그래서 L1 Norm은 Feature Selection 이 가능하여 Sparse Model에 적합하다.  

![weightdecay](http://androidkt.com/wp-content/uploads/2021/09/L1-Regula.png)

그 결과로

- Overfitting을 방지한다.
- Weight를 작게 유지해서 Gradient Exploding을 방지한다.

<strong>Zero Initialization</strong>

결론적으로 가중치를 0으로 초기화시키는 방법은 아주 나쁜 방법이다. 0으로 초기화하면 제대로 학습되지 않는다. 그 이유는 각 뉴런이 training중에 동일한 feature를 학습하기 때문이다. 모두 동일한 feature을 학습한다는 것은 역전파 과정에서 모든 weight 값이 동일하게 바뀌어 학습이 진행되지 않는 것을 의미한다.

가중치가 고르게 되는 대칭적 가중치 문제를 해결하기 위해서 가중치를 무작위로 설정하는 방법이 있다.

<strong>Random Initialization</strong>

이 방법도 zero initialization 에서 설명한 것과 같이 같은 가중치 값을 가져서 뉴런을 여러 개 둔 의미가 사라지는 문제가 발생한다. 
 
결론적으로 기울기 문제를 막기 위해서는 다음과 같은 결론을 내릴 수 있다.

- activation의 평균은 0이어야 한다.
- activation의 variance는 모든 layer에서 동일하게 유지되어야 한다.



<strong>Xavier Initialization</strong>

각 레이어의 활성화 값을 더 광범위하게 분포시킬 목적으로 weight의 적절한 분포를 찾으로 했다. tanh 또는 sigmoid로 활성화 되는 초깃값을 위해 이 방법을 주로 사용한다. 이전 layer의 neuron의 개수가 n이라면 표준편차가 $\dfrac{1}{\sqrt{n}}$ 인 분포를 사용한다. 

너무 크지도 않고 작지도 않은 가중치를 사용하여 기울기가 소실되거나 폭발하는 문제를 막는다.

> ReLU를 이용할 때는 비선형성이 발생하여 Xavier 초기화가 효과적이지 않는다.

<strong>He Initialization</strong>

He 초기화에선느 표준편차가 $\dfrac{2}{\sqrt{n}}$ 인 정규분포를 사용한다. ReLU는 음의 영역이 0이라서 활성화되는 영역을 더 넓게 분포시키기 위해서 2배의 계수가 필요하다고 간단하게 해석할 수 있다.

> activation function으로 ReLU를 사용하면 He 초깃값을 쓰는 것이 좋다.


### Small Learning Rate

너무 학습률이 크면 확률적 경사 하강법을 실시하다가 Gradient Exploding 문제가 발생할 수 있다. 적당한 Learning Rate를 설정하면 문제를 해결할 수 있다.

### Batch Normalization

학습 과정에서 각 배치 단위별로 데이터가 다양한 분포를 가지더라도 각 배치별로 "평균", "분산"을 이용해 정규화하는 것을 뜻한다. 학습 과정에서는 평균과 분산을 각 배치마다 구해서 정규화를 진행하고 추론 과정에서는 학습 과정에서 구한 평균과 분산을 이용하여 정규화를 진행한다.

<strong>Pytorch</strong>

```python
torch.nn.BatchNorm1d(num_features)
torch.nn.BatchNorm2d(num_features)
```

-   BatchNorm1d
    Input 과 Output이 (N, C) 또는 (N, C, L) 의 형태를 가진다.

-   BatchNorm2d
    Input 과 OUtput이 (N, C, H, W) 의 형태를 가진다.
    C는 Channel을 말한다.


> Batch Normalization Layer 는 활성화 함수 앞에 사용한다.


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

![2d Convolution with 3D input]

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






## Reference



- CNN

    - [https://ydy8989.github.io/2021-02-02-conv/](https://ydy8989.github.io/2021-02-02-conv/)
    - [https://www.youtube.com/watch?v=KuXjwB4LzSA&t=11s](https://www.youtube.com/watch?v=KuXjwB4LzSA&t=11s)
    - [https://89douner.tistory.com/57](https://89douner.tistory.com/57)
    - [https://yjjo.tistory.com/8](https://yjjo.tistory.com/8)
    - [https://github.com/ifding/learning-notes/blob/master/machine-learning/1d-2d-and-3d-convolutions-in-cnn.md](https://github.com/ifding/learning-notes/blob/master/machine-learning/1d-2d-and-3d-convolutions-in-cnn.md)
