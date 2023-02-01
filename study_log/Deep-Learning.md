# ETC...  

## Index

- [ETC...](#etc)
  - [Index](#index)
  - [Regularization](#regularization)
    - [Problem](#problem)
    - [Solution](#solution)
    - [Change Activation Function](#change-activation-function)
    - [Careful Initialization](#careful-initialization)
    - [Small Learning Rate](#small-learning-rate)
    - [Batch Normalization](#batch-normalization)
    - [Dropout](#dropout)
  - [Optimization](#optimization)
    - [SGD (Stochastic Gradient Descent; 확률적 경사하강법)](#sgd-stochastic-gradient-descent-확률적-경사하강법)
    - [Momentum](#momentum)
    - [Adagrad](#adagrad)
    - [RMSProp](#rmsprop)
    - [Adam (Adaptive Moment Estimation)](#adam-adaptive-moment-estimation)
  - [Activation](#activation)
    - [ReLU](#relu)
  - [Trainig Refinements](#trainig-refinements)
    - [Mixup Training](#mixup-training)
    - [Label Smoothing](#label-smoothing)
  - [Reference](#reference)

---


<a name="regularization"></a>
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


<a name="weightinit"></a>
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


<a name="batchnorm"></a>
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

<a name="dropout"></a>
### Dropout

Drop-out은 서로 연결된 연결망에서 0부터 1사이의 확률로 뉴런을 제거하는 기법이다. Drop-out 이전에 4개의 뉴런끼리 모두 연결되어 있는 전결합 계층에서 4개의 뉴런은 각각 0.5의 확률로 제거될지 말지 랜덤하게 결정된다.    

그렇다면 왜 사용할까??

Drop-out은 어떤 특정한 설명 변수의 Feature만을 과도하게 집중하여 학습함으로써 발생할 수 있는 과대적합을 방지하기 위해 사용된다. 드롭아웃을 적용하여 더욱 편향되지 않은 출력값을 얻는 데 효과적이다. 


<a name="optimization"></a>
## Optimization

**Optimizer**

손실함수를 줄여나가면서 학습하는 방법은 어던 옵티마이저를 사용하느냐에 따라 달라진다. 경사하강법은 가장 기본적이지만 가장 많이 사용되는 최적화 알고리즘이다. 손실 함수의 1차 도함수에 의존하는 first-order 최적화 알고리즘으로 함수가 어떤 방향으로 가중치를 업데이터해야 하는지를 계산한다. 역전파를 통해 손실이 한 계층에서 다른 계층으로 전달되고, 다시 이 손실에 따라 모델의 파라미터가 수정되어 손실을 최소화할 수 있다.

경사 하강법의 문제점

- 한번 학습할 때마다 모든 데이터셋을 이용한다. -> 확률적 경사 하강법
- 학습률 정하기: 학습률이 너무 크다면, 최솟값을 계산하도록 수렴하지 못하고 손실값이 계속 커지는 방향으로 진행될 수도 있다. 학습률이 너무 작다면, 최솟값을 찾는데 오랜 시간이 걸린다.
- Local Minima: 진짜 목표인 global minimum을 찾지 못하고 local minimum에 갇혀버릴 수도 있다. 
- 메모리 한계: 모든 데이터를 한 번에 다 학습한다면 메모리의 용량이 부족할 수 있다.

=> SGD 등장!


<a name="sgd"></a>
### SGD (Stochastic Gradient Descent; 확률적 경사하강법)

SGD는 GD와 유사하지만 전체 데이터가 아닌 미니 배치 사이즈만큼의 데이터로 경사 하강법을 실시한다는 차이가 있다. 이를 통해 학습 속도를 빠르게 할 수 있을 뿐만 아니라 메모리도 절약할 수 있다. 

=> 학습률 설정, local minima 문제, oscillation 문제 해결 x


<a name="momentum"></a>
### Momentum

모멘텀은 SGD의 높은 편차를 줄이고 수렴을 부드럽게 하기 위해 고안되었다. 이는 관련 방향으로의 수렴을 가속화하고 관련 없는 방향으로의 변동을 줄여준다. 말 그래도 이동하는 방향으로 나아가는 '관성'을 주는 것이다. 

![momentum](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbzwTc0%2FbtrghQoeWvu%2FRvKTkqI4ObkPQreXTBqgUk%2Fimg.png)


- SGD에 비해 파라미터의 분산이 줄어들고 덜 oscillate 한다는 장점이 있고, 빠르게 수렴한다.
- r라는 새로운 하이퍼 파라미터를 잘 설정해줘야 한다는 단점이 있다.


<a name="adagrad"></a>
### Adagrad

지금까지의 옵티마이저의 단점 중 하나는 학습률이 모든 파라미터와 각 cycle에 대해 일정하다는 것이다. Adagrad는 각 파라미터와 각 단계마다 학습률을 변경할 수 있다. 

> 이 알고리즘의 기본적인 아이디어는 ‘지금까지 많이 변화하지 않은 변수들은 step size를 크게 하고, 지금까지 많이 변화했던 변수들은 step size를 작게 하자’ 라는 것이다. 자주 등장하거나 변화를 많이 한 변수들의 경우 optimum에 가까이 있을 확률이 높기 때문에 작은 크기로 이동하면서 세밀한 값을 조정하고, 적게 변화한 변수들은 optimum 값에 도달하기 위해서는 많이 이동해야할 확률이 높기 때문에 먼저 빠르게 loss 값을 줄이는 방향으로 이동하려는 방식이라고 생각할 수 있겠다.
 - 출처 : http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html

- 이계도 함수를 계산해야 하기 때문에 계산 비용이 많이 든다.
- 학습을 진행하면서 학습률이 줄어든다는 문제점이 있다.
- 최솟값에 도달하기도 전에 학습률이 0에 수렴해버릴 수도 있다.


<a name="rmsprop"></a>
### RMSProp

RMSProp은 Adagrad에서의 단점을 해결하기 위해서 지수 이동평균을 이용한다. 지수 이동평균을 이용해 가중치로 영향력을 decay한다. 


<a name="adam"></a>
### Adam (Adaptive Moment Estimation)

각 파라미터마다 다른 크기의 업데이트를 진행하는 방법이다. Adam의 직관은 local minima를 뛰어넘을 수 있다는 이유만으로 빨리 굴러가는 것이 아닌, minima의 탐색을 위해 조심스럽게 속도를 줄이고자 하는 것이다. 


<a name="activation"></a>
## Activation


<a name="relu"></a>
### ReLU

비선형성을 증가시킨다?? 증가시키면 뭐가 좋다구?

- 합성곱 계층 개수가 커지면, 그리고 각 합성곱 계층 다음에 ReLU와 같은 '비선형' 활성화 함수가 오면 네트워크가 복잡한 특징을 학습할 수 있는 능력이 증대된다. **(더 많은 비선형 연산을 결합함으로써)**
- ReLU의 장점
  - Sparse Activation: 0이하의 입력에 대해 0을 출력함으로서 부분적으로 활성화할 수 있다.
  - Efficient Gradient Propagation: Gradient의 Vanishing이 없으며 Gradient가 Exploding 되지 않는다. (기울기가 0<x에서 1이기 때문이다!)
  - Efficient Computation: 선형 함수이므로 미분 계산이 매우 간단하다.

- 비선형 함수를 활성화 함수로 이용하는 이유는 선형함수를 사용할 시 층을 깊게 하는 의미가 줄어들기 때문이다. 뉴럴 네트워크에서 층을 쌓는 효과를 얻고 싶다면 반드시 비선형 함수를 사용해야 한다.
- 비선형성이 증가한다는 것은 그만큼 복잡한 패턴을 좀 더 잘 인식할 수 있게 된다는 의미이다.!

## Trainig Refinements

### Mixup Training

학습을 진행할 때 랜덤하게 두 개의 샘플 $(x_i, y_i)$와 $(x_j, y_j)$를 뽑아서 $(\widehat{x}, \widehat{y})$를 만들어서 학습에 사용한다. 

$\hat{x} = \lambda x_i + (1-\lambda) x_j$

$\hat{y} = \lambda y_i + (1-\lambda) y_j$

$\lambda \in [0, 1]$

![mixup_training](https://crazyoscarchang.github.io/images/2020-09-27-revisiting-mixup/classical_mixup.png)

### Label Smoothing

정답 클래스에 대해서 완전히 confident하게 1의 값을 부여하는 것이 아니라 나머지 라벨에도 약간의 값을 [0.9, 0.05, 0.05]와 같이 부여해서 너무 한 클래스에 over confident하지 않도록 label smoothing 기법을 사용한다. 



![label_smoothing](https://blog.kakaocdn.net/dn/uO0Hr/btqFFVzoC6h/N1cMUSyijCh6fvwBm64nh0/img.png)


## Reference

- Regularization
  - [https://heytech.tistory.com/127](https://heytech.tistory.com/127)
  - [https://heytech.tistory.com/125](https://heytech.tistory.com/125)


- Optimization

    - [https://heeya-stupidbutstudying.tistory.com/entry/ML-%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%97%90%EC%84%9C%EC%9D%98-Optimizer-%EC%97%AD%ED%95%A0%EA%B3%BC-%EC%A2%85%EB%A5%98](https://heeya-stupidbutstudying.tistory.com/entry/ML-%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%97%90%EC%84%9C%EC%9D%98-Optimizer-%EC%97%AD%ED%95%A0%EA%B3%BC-%EC%A2%85%EB%A5%98)
    
- Activation
    
    - [https://driip.me/af3d08bb-c39a-416b-9301-51594a8f6848](https://driip.me/af3d08bb-c39a-416b-9301-51594a8f6848)