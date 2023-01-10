# NLP

## Index

- [NLP](#nlp)
  - [Index](#index)
  - [RNN](#rnn)
  - [LSTM \& GRU](#lstm--gru)
  - [Seq2Seq \& 1D CNN \& Bi-LSTM](#seq2seq--1d-cnn--bi-lstm)
  - [Embedding Layer](#embedding-layer)
    - [텍스트 임베딩](#텍스트-임베딩)
    - [이미지 임베딩](#이미지-임베딩)
  - [Reference](#reference)

--- 

## RNN

## LSTM & GRU

## Seq2Seq & 1D CNN & Bi-LSTM



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

  

## Reference
