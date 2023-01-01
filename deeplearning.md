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

