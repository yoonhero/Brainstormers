# Transformer


## Table of Contents

<!-- TOC -->

- [Transformer](#transformer)
    - [Table of Contents](#table-of-contents)
    - [Attention](#attention)
        - [Self Attention](#self-attention)

<!-- /TOC -->

## Attention

Machine Translation **Seq2Seq**

- context 벡터만을 이용하면 bottleneck 현상이 일어날 수 있다.
- Vanishing / Exploding Gradient
- 단어 생성할 때마다 context 모두 사용 (선택적 집중이 필요하다.)

### Self Attention

Self Attention이란 자기 자신에 대한 Attention을 말한다. 왜 이러한 방법을 사용해야할까? 기존의 RNN이나 LSTM 기반의 seq2seq machine translation의 문제점 중 하나인 단어를 생성할 때마다 모든 context를 사용하는 점을 해결하기 위해서이다. 우리가 글을 읽을 때 모든 글자에 집중해서 읽는 것은 아니다. 만약에 "어제 카페에 갔었어 거기 사람 많더라" 라는 문장이 있다고 해보자. 우리가 "갔었어"라는 단어를 읽을 때 "카페에" "어제" 이런 단어에 **선택적**집중을 한다. Transformer에서는 기존의 RNN과 같은 방식을 아예 없애고 오로지 self attention만으로 네트워크를 구성했다. 이 결과로 엄청난 성능으로 각종 분야에서 SOTA가 되었으며 Vision분야에서도 Vision Transformer가 나오는 등 2010년 후기에 나온 가장 혁명적인 아키텍처라고 할 수 있다. 

- 모든 시퀀스를 볼 수 있다. (1D-CNN의 커널 사이즈 문제를 해결할 수 있었다.)
- 멀리 있는 정보도 손실되지 않는다. (RNN의 context 소실 문제 해결할 수 있었다.)

$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

- Query: 디코더의 이전 레이어 hidden state
- Key: 인코더의 output score, 영향을 주는 인코더의 토큰들
- Value: 인코더의 output score, 그 영향에 대한 가중치가 곱해질 인코더의 토큰들

쿼리와 키를 행렬곱하고 이 값을 $\sqrt{d_k}$로 나누어주고 소프트맥스를 취해서 스코어를 구한다. 이 행렬에 밸류를 곱해서 어텐션 연산을 마친다.


${QK^T}$을 통해서 쿼리 벡터와 키 벡터 사이의 문맥적 관계성이 녹아든 결과를 얻을 수 있다. 

$softmax(\frac{QK^T}{\sqrt{d_k}})$를 통해서 나온 값을 밸류 벡터와 행렬곱해서 최종적인 어텐션 스코어를 얻을 수 있다. 

여기서 $\sqrt{d_k}$로 나누어주는 이유는 Attention Score 유사도를 구하는 방법이 3가지가 있는데, Dot Product, Scaled Dot Product, Weighted Dot Product가 있다.

- Dot Product: $QK^T$
- Scaled Dot Product: $\frac{QK^T}{\sqrt{d_i}}$
- Weighted Dot Product: $QK^T W_i$


### Multi-Head Attention

Multi-Head Attention이란 Self Attention을 여러번 수행한 것을 가르킨다. 이는 곧 여러 헤드가 셀프 어텐션을 계산한다는 것이다. 
여러 독자가 글을 읽으면 글에 대한 다양한 생각이 있을 수 있는 것처럼 여러 헤드가 한 sequence를 모두 독자적으로 attention을 계산함으로서 셀프 어텐션을 보완할 수 있다. 


### Masked Multi-Head Attention




