# NLP

## Index

- [NLP](#nlp)
  - [Index](#index)
  - [RNN](#rnn)
    - [First-Order System](#first-order-system)
    - [State-Space Model as RNN](#state-space-model-as-rnn)
    - [RNN: Problem Type](#rnn-problem-type)
  - [LSTM \& GRU](#lstm--gru)
    - [RNN의 문제점](#rnn의-문제점)
    - [LSTM](#lstm)
    - [GRU](#gru)
    - [GRU vs LSTM](#gru-vs-lstm)
  - [Seq2Seq](#seq2seq)
    - [Seq2Seq 의 훈련법 - Teacher Force Training](#seq2seq-의-훈련법---teacher-force-training)
  - [1D CNN](#1d-cnn)
  - [Bi-LSTM](#bi-lstm)
  - [One-hot Encoding](#one-hot-encoding)
  - [Embedding Layer](#embedding-layer)
    - [Word2Vec](#word2vec)
    - [GloVe](#glove)
    - [텍스트 임베딩](#텍스트-임베딩)
    - [이미지 임베딩](#이미지-임베딩)
  - [Reference](#reference)

--- 

## RNN

<strong>Recurrent Neural Network</strong>는 시계열 데이터를 처리하기 적합한 인공지능 신경망 구조이다. 여기서 시계열 데이터란 시간에 따라서 데이터가 바뀌는 형식의 데이터를 말한다. RNN은 Computer Vision Task에서 쓰이는 CNN과도 유사성이 있다.

- CNN은 이미지 구역별로 같은 Weight를 공유한다.
- RNN은 시간별로 같은 Weight를 공유한다.

시간별로 같은 Weight를 공유한다는 말은 CNN의 동일한 커널이 이미지 전체에 합성곱 연산을 하는 것처럼 RNN도 시간별로 같은 값을 공유한다는 것이다. 

### First-Order System

현재 시간의 상태가 이전 시간의 상태와 관련이 있다고 가정하는 것이다. ex) 시계열 데이터

$x_t = f(x_{t-1})$

- $x_t$ : 현재 상태
- $x_{t-1}$ : 전 상태
- $f()$ : First Order System

만약 현재 시간 상태가 이전 시간 상태와 현재의 입력과 관계가 있을 경우에는

$x_t = f(x_{t-1}, u_t)$

- $u_t$ : 현재의 입력

### State-Space Model as RNN

1차원 시스템의 모형: $x_t = f(x_{t-1}, u_t)$

> Q: 모든 시간 t에서 모든 상태 $x_t$ 관측 가능할까?

이 질문에 답은 모든 상태를 관측할 수는 없다는 것이다. 날씨 예측에서는 모든 상태를 예측하지 못하고 일부만 관측할 수 있는 것처럼 모든 상태를 관측할 수 없어서 위 수식에서 관측 가능한 상태만 모여서 출력한다.

- $x_t = f(x_{t-1}, u_t)$
- $y_t = h(x_t)$

여기서 함수 $f, h$를 근사하기 위해서 Neural Network 를 사용하는 것이다. 

RNN은 최종 출력층에서 ANN을 FC하여 ${\hat{y_{t}}}$ 을 출력하는 형태가 많이 사용된다. 마치 CNN에서 Convolutional Layer와 Max Pooling 층을 통과하여 최종적으로 FC된 ANN으로 라벨을 출력하는 형태처럼 RNN이 시계열 데이터에서 시간적인 관계를 포함하여 의미있는 특성 값으로 Encoding하면 이를 ANN이 Decoding하는 것이라고 할 수도 있다. 

![rnn](https://miro.medium.com/max/1400/1*HgAY1lLMYSANqtgTgwWeXQ.png)

### RNN: Problem Type

- Many-to-Many: 번역 등
- Many-to-One: 예측 등
- One-to-Many: 생성 등

![rnns](https://iq.opengenus.org/content/images/2020/01/export.png)


## LSTM & GRU

### RNN의 문제점

RNN에서의 가장 큰 문제점은 Exploding Gradient와 Vanishing Gradient이다. 과거의 상태가 RNN 레이어를 거치면 거칠 수록 곱해지는 값이 1보다 크면 폭발하게 되고 1보다 작으면 0으로 수렴하여 사라지는 문제가 있다. 이러한 문제점은 훈련이 끝날 때까지 알지 못해서 수정에도 어려움이 있다. 이 문제를 해결하기 위해 Gradient Clipping 같은 해결법이 나왔지만 궁극적으로 새로운 네트워크 구조가 필요했고 이로 인해서 Gated RNNs인 LSTM과 GRU가 등장하게 되었다.

### LSTM

<strong>Long Short Term Memory</strong> 은 RNN 구조에 Gradient Flow를 제어할 수 있는 "밸브" 역할이 추가된 구조라고 생각하면 된다. 

> LSTM = RNN + Gate 구조 (4개의 MLP 사용)

![lstm](https://miro.medium.com/max/984/1*Mb_L_slY9rjMr8-IADHvwg.png)

- 1. Input Gate & Forget Gate
  
    이 Gate에서는 '새로운 입력과 이전 상태를 참고해서 어느정도의 비율로 $x_{t}$와 $h_{t-1}$을 사용할까?' 를 정한다.

- 2. Cell

    Forget Gate와 Input Gate에서 출력된 값을 적당히 섞는다.

- 3. Output Layer

    Forget Gate, Input Gate, Cell의 출력된 값을 모두 입력값으로 사용하여 최종적인 결과값을 출력한다.

> LSTM 구조 너무 복잡해;;; ====> GRU 등장


```python
nn.RNN(input_size hidden_size num_layers batch_fist)
```

- input_size: 입력해 주는 특성 값의 개수이다. (batch_size, time_steps, input_size)
- hidden_size: hidden state 개수를 지정한다.
- num_layers: RNN층을 쌓아 올릴 수 있다.
- batch_first: 첫번째 차원을 배치 차원으로 사용할지 

### GRU

<strong>GRU</strong>는 LSTM 구조에서 cell 구조가 제외되어 간소화된 구조이다.

![GRU](https://cdn-images-1.medium.com/freeze/max/1000/1*GSZ0ZQZPvcWmTVatAeOiIw.png?q=20)

### GRU vs LSTM

- Training 시에 파라미터 수가 GRU가 LSTM보다 적어서 GRU는 학습이 빠르다.
- 두 신경망의 성능은 Task에 따라서 천차만별이다.
  - LSTM or GRU 를 실험해고 테스트 결과에 따라서 사용하면 된다. (두 신경망 모두 RNN보다 월등히 성능이 좋다.)

## Seq2Seq 

RNN을 번역 Task에 활용한다고 가정을 해보자. "I am a student"와 같은 문장이 입력이고 "저는 학생입니다"가 출력이라고 해보자. "I"가 입력값으로 들어가면 "저는"을 출력해야 한다. 하지만 이렇게 번역을 한다면 앞의 값만 보고 번역을 진행해서 언어 간의 어순이 다를 때 번역이 잘 되지 않는다. 이런 문제를 해결하기 위해서 <strong>Seq2Seq</strong> 신경망 구조가 제안되게 되었다.

![seq2seq](https://wikidocs.net/images/page/24996/%EC%9D%B8%EC%BD%94%EB%8D%94%EB%94%94%EC%BD%94%EB%8D%94%EB%AA%A8%EB%8D%B8.PNG)

전체 신경망은 Encoder RNN(LSTM)과 Decoder RNN(LSTM)으로 구성되어 있다.

<strong>Seq2Seq</strong>가 단일 RNN과 가장 큰 차이점은 바로 Encoder Layer가 전체의 문맥을 하나의 Context Vector에 담아서 Decoder Layer에 전달해준다는 것이다. 이를 통해서 RNN이 자신의 앞의 값만 참고할 수 있는 문제를 해결할 수 있는 것이다. 인코더에서는 전체 단어를 하나씩 게이트에 통과시켜서 $h_t$ 를 만들고 이는 문장 전체의 의미를 담고 있어서 Context Vector라고 할 수 있다. (인코더에서 각 LSTM을 통과하여 도출된 출력값을 사용하지 않는다.) 전달된 Context Vectore를 Hidden State로 사용하여 <시작>과 같은 태그를 주면 출력을 시작한다. 각 단계에서 출력된 값은 다음 단계의 입력으로 사용되고 <끝> 과 같은 문장의 끝을 알려주는 토큰이 출력될 때까지 이 과정을 반복하는 것이다.

### Seq2Seq 의 훈련법 - Teacher Force Training

훈련 초기에는 랜덤한 Context Vector을 디코더에 전달하기 때문에 디코더의 출력은 굉장히 이상할 수 있다. 이 출력을 바탕으로 다음 단어를 예측하도록 신경망을 지도학습 시키면 성능에 엄청난 영향을 줄 것이다. 그래서 Seq2Seq 모델을 훈련시킬 때에는 Teacher Force Training을 사용하여 훈련시에는 디코더의 입력을 모두 알려줘서 올바르게 훈련될 수 있도록 한다. 

> 이러한 근본적인 문제 해결을 위해서 Attention 기법이 등장하게 되었다.


## 1D CNN

흔히 Image Processing Task 에서 흔하게 사용되는 CNN은 공간적으로 공통적인 Weight를 사용하여 특징을 추출한다는 특성이 있다. 이는 RNN이 시간적으로 공통적인 Weight를 사용하여 특징을 추출하는 것과 굉장히 유사한다. NLP에서는 자연어 처리용 CNN으로 <strong>1D CNN</strong>이라는 신경망이 사용된다. 

사용예시 => Word2Vec와 1d conv filter을 활용하여 NLP를 수행할 수 있다.

![1dcnn](https://ars.els-cdn.com/content/image/1-s2.0-S0888327020307846-gr5.jpg)

## Bi-LSTM

LSTM이 단방향인 점을 보완하여 양방향으로 진행할 수 있는 신경망 구조이다. 이 신경망에는 LSTM에서 과거의 값만으로 현재의 값을 예측하는 것과 다르게 미래의 값까지 같이 학습에 사용하여 현재의 값을 예측하여 성능이 더 높아질 수 있다.

![bilstm](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-25_at_8.54.27_PM.png)


## One-hot Encoding

컴퓨터 또는 기계는 문자보다는 숫자를 더 잘 처리할 수 있다. 이를 위해 자연어 처리에서는 문자를 숫자로 바꾸는 여러가지 기법들이 존재한다. 원-핫 인코딩은 많은 기법 중에서 단어를 표현하는 가장 기본적인 표현 방법이다.

- One-hot Encoding
	- 필요한 정보를 담은 하나의 값만 1로 두고, 나머지 값은 0으로 설정한다. 
	- 대부분의 값이 0을 갖고 단 한 개의 1인 값을  가지는 일종의 Sparse Matrix (희소 행렬)로 표현된다.
	- 단어가 많아질수록 벡터 공간만 커지는 비효율적인 방법이다.
	- 원-핫 인코딩은 단어가 무엇을 의미하는지 설명하지 못한다.
	- 단어간의 유사성을 계산할 수 없다.

-> 이런 문제가 있었기에 문자를 Dense Matrix로 변환하는 표현법이 제시되었다.

## Embedding Layer

![embedding](./docs/torch-embedding.jpeg)
 
우리가 사용하는 언어나 이미지는 0과 1로만 이루어진 컴퓨터 입장에서 그 의미를 파악하기가 어렵다. 예를 들어 우리가 인공지능 챗봇을 제작한다고 하자. 우리가 입력한 말을 과연 컴퓨터가 바로 이해할 수 있을까? 우리는 우리가 입력한 말을 tokenize하는 과정을 통해서 '언어의 벡터화'를 한다. 이런 tokenize하는 일련의 과정을 **Word Embedding**이라고 한다.

사람이 사용하는 언아나 이미지를 컴퓨터에게 이해시키기 위해서는 어떤 **벡터 공간**에 우리가 표현하고자 하는 정보를 mapping해야 한다.

고차원의 정보를 저차원으로 변환하면서 필요한 정보를 보존하는 것을 **임베딩(Embedding)** 이라고 한다. 


희소 표현인 원-핫 인코딩은 단어 집합의 크기가 커질수록 벡터의 차원 또한 커지기에 비효율적이고 단어간의 유사도를 나타낼 수 없다. 이러한 희소 표현과 반대되는 표현을 바로 밀집 표현이라고 한다. Word Embedding도 이 밀집 표현으로 밀집 표현은 사용자가 설정한 값으로 단어 벡터 표현의 차원을 맞춰서 벡터의 차원이 조밀해졌다고 하여 Dense Vector라고도 한다. 


워드 임베딩 방법론

- LSA
- Word2Vec
- FastText
- Glove

### Word2Vec

워드투벡터는 밀집 표현의 한 종류인 분산 표현 방법을 사용한다. 분산 표현은 분포 가설이라는 가정하에 만들어진 표현 방법이다. 이 가정은 **비슷한 위치에서 등장한느 단어들은 비슷한 의미를 가진다**라는 가정이다. 이 가정을 토대로 단어의 의미를 여러 차원에다가 분산하여 표현하여 단어간의 유사도를 계산할 수 있는 것이다.

- CBOW(Continuous Bag Of Words) 
  - CBOW는 주변에 있는 단어들을 가지고, 중간에 있는 단어들을 예측하는 방법이다.  

- Skip-Gram
  - Skip-Gram은 중간에 있는 단어로 주변 단어들을 예측하는 방법이다.

### GloVe

워드투벡터와 함께 문장을 임베딩할 때 많이 사용된다고 한다. 성능은 비슷하여 실제로 테스트해보고 무엇을 사용할지 정하면 된다고 한다. 

### 텍스트 임베딩

- 임베딩 벡터
	- 밀집행렬로 임베딩된 벡터는 각 요소에서 단어의 서로 다른 특성을 나타낸다.
	- 각 요소에는 단어가 관련 특성을 대표하는 정도를 나타내는 0~1 사이의 값이 포함된다.
	- 이런 임베딩을 통해 텍스트를 단순히 '구분' 하는 것이 아닌 의미적으로 '정의'하는 것이라고 볼 수 있다.
	 ![text-embedding](https://velog.velcdn.com/images%2Fdongho5041%2Fpost%2F6cff5fbf-2a1c-42c1-8a08-613c4582729d%2Fimage.png)

파이토치에서 제공하는 nn.Embedding() 레이어는 단어를 랜덤한 값을 가지는 밀집 벡터로 벼환한 뒤에, 인공 신경망의 가중치를 학습하는 것과 같은 방식으로 단어 벡터를 학습하는 방법을 사용한다.

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





## Reference
