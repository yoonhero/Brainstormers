# Transformer


## Table of Contents

<!-- TOC -->

- [Transformer](#transformer)
  - [Table of Contents](#table-of-contents)
  - [Attention](#attention)
    - [Self Attention](#self-attention)
    - [Multi-Head Attention](#multi-head-attention)
    - [Masked Multi-Head Attention](#masked-multi-head-attention)
  - [Vision Transformer](#vision-transformer)

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

```python
wei = torch.zeros((8, 8))
wei = wei.masked_fill(tril==0, float("-inf"))
wei = F.softmax(wei, dim=-1)
wei
```

```bash
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
```

Masked Multi-Head Attention은 트랜스포머의 디코더나 GPT에서 사용되는 어텐션 기법 중 하나이다. 트랜스포머의 Self Attention을 디코더에서 사용하면 다음 단어의 확률 분포를 예측해야하는 디코더가 정답을 보면서 학습을 진행하기 때문에 학습이 잘 이루어지지 않을 수 있다. 이 문제를 해결하기 위해서 한 가지 트릭을 사용한다. 바로 예측하려는 다음 timestep의 정보를 보이지 않도록 masking하는 것이다. 선형대수적으로 접근하면 하삼각행렬에 있는 값만을 softmax를 통과했을 때 살려두기 위해서 나머지 값들을 -inf로 맞추게 된다. 이를 통해서 전 문맥만 보고 다음 단어를 예측해야하는 GPT와 같은 모델이 잘 학습될 수 있도록 할 수 있다.


## Vision Transformer

NLP에서 SOTA의 성능을 보이던 Transformer이 Vision 분야에서도 뛰어난 성능을 보일 수 있다는 것을 보인 모델이다. CNN을 하나도 사용하지 않고 Attention만으로 엄청난 모델을 설계했다는 것에 의의가 있다.

하지만 VIT는 translation equivariance 및 locally 와 같은 CNN 고유의 inductive bias(보지 못한 입력에 대한 출력을 예측할 때 사용되는 가설)이 없기 때문에 더 많은 데이터가 필요하다고 한다. 구글은 이를 3억장의 사진에 의해서 pretrained하여 해결했다고 한다.

![vit](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FI6CZv%2Fbtq4W1uStWT%2FBBBI8YYnbCgfO8rKeZTK31%2Fimg.png)

전체적인 모델의 로직은 Transformer의 엔코더와 유사하다. 여기서 주목할 점은 바로 전체 이미지를 바로 사용하는 것이 아닌 이미지를 패치단위로 나누어서 예측을 진행했다는 것이다.

트랜스포머와 마찬가지로 패치마다 Postional Embedding을 더해주고 Encoding Block을 여러번 통과하여 예측을 진행한다.

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  # Embedding dim으로 변환하며 패치크기의 커널로 패치크기만큼 이동하여 이미지를 패치로 분할 할 수 있음.

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # 세번째 차원부터 끝까지 flatten (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x
```


## ELECTRA

>> ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)

기존의 BERT 모델은 입력 중 일부를 [MASK] 토큰으로 치환하고 이를 복원하는 형태로 학습을 진행했다. 하지만 이런 모델을 훈련시키기 위해서는 너무나도 많은 컴퓨팅 자원이 필요하다. 그래서 본 연구진은 RTD(Replaced Token Detection)이라는 새로운 사전 학습법을 제안해서 보다 빠르고 정확한 모델을 개발할 수 있었다.

**RTD**

- 1. MASK된 토큰 부분을 Generator가 바꾼다.
- 2. Discriminator이 Generator가 바꾼 부분인지 판단한다. Replaced or Original 이진분류를 학습한다.
- 3. 이렇게 학습을 진행해 나간다!

판별자와 생성자 모두 트랜스포머 인코더 구조를 사용한다.

**Weight Sharing**

Discriminator와 Generator 모두 트랜스포머 구조를 활용했다. 실험자는 임베딩의 가중치를 공유할 때와 전체 가중치를 공유할 때의 성능을 비교했다. 실험 결과 전체 가중치를 공유할 때의 성능이 가장 뛰어났다. 이는 바로 Discriminator은 학습시에 들어오는 입력에 대해서만 학습을 진행하지만 Generator는 들어온 입력에 대해서 출력을 하면서 Softmax를 취할 때 전체의 의미를 함축해서 학습을 진행할 수 있기 때문이다.


**Smaller Generators**

ELECTRA에서는 BERT와 유사한 크기의 트랜스포머 2개를 사용하기에 단순 계산만으로도 2배의 컴퓨팅 자원이 필요하다는 것을 알 수 있다. 이는 너무 많은 컴퓨팅 자원을 필요로 하기에 Smaller Generator을 실험해봤고 좋은 결과를 얻을 수 있었다고 한다. Discriminator과 Generator의 크기가 유사하면 Discriminator가 너무 강력해져서 Discriminator을 훈련시키기 어려울 수 있다고 한다. 또 판별자가 실제 데이터의 분포가 아닌 generator을 모델링하는 데에 사용될 수 있다고 한다.

**Maximum Likelihood & Adversarial Traning**

Maximum likelihood aims to accomplish to lower negative log loss by backpropagation. We can use maximum likelihood objective function when we try to maximize the likelihood generating data such as image classifcation. But adversarial traning aims to lower the combination loss and the adversarial loss. This training method was introduced in the well-known paper "GAN". Adversarial Training can prompt to correctly classify between generated image and sample images.

>> 결론: 생성자가 생성한 토큰을 판별자가 판단하도록 하여서 두 네트워크를 적대적으로 훈련시켰을 때, 작은 크기의 모델이라도 좋은 성능을 보일 수 있었다.


## Reformer

일반적인 트랜스포머 모델의 문제는 Self-Attention과 FF 레이어를 통과할 때 발생한다. Self-Attention 연산에서는 Key와 Query의 행렬곱 연산을 수행하는데 이때 Key-Query는 데이터 열와 유사하기에 입력 토큰의 개수가 많아질수록 연산량이 기하급수적으로 늘어난다. 또한 FF 레이어를 사용하면서 차지하는 메모리의 용량도 무시할 수 없다. 역전파 과정에서도 이러한 출력값을 저장하고 있어야 학습이 진행되기에 학습 시의 메모리 용량도 무시할 수 없다.


**Contribution**

- LSH을 사용하여 Self-Attention의 문제점을 해결한다.
- Chunking을 활용하여 FF의 문제점을 해결한다.
- Reversible Layer을 활용하여 Residual Connection에서의 연산량을 줄인다.


### Locality-Sensitive Hashing

Hashing은 임의의 데이터를 길이가 해시 값으로 치환하는 것을 의미한다. 보통 Hash 값은 연결된 데이터와 전혀 관련이 없을 때가 많다. 그래서 상대적 위치를 확인하거나 다른 데이터를 찾는  데이터에 대한 비교 분석을 할 때 실제 데이터값을 비교하는 연산이 필요하다. 이때 가까운 데이터끼리는 가까운 Hash 값을 갖도록 구성할 수 있따면 비교하는 연산을 Hash 값에 대한 연산으로 근사하여 연산량을 줄일 수 있다. 이러한 Hashing 방법을 Locality-Sensitive Hashing이라고 한다.

1. 전체 데이터 포인트들의 벡터를 단위 구면에 사상한다. 이렇게 되면 전체 데이터 포인터를 각도만을 사용하여 기술할 수 있다.
2. 비슷한 데이터들은 같은 사분면에 있어서 사분면의 번호를 Hash 값으로 사용한다면 비슷한 데이터를 가깝게 구성할 수 있다.
3. 이 구면을 필요한 만큼 임의로 회전시키고 데이터가 가까우면 전체 Hash 값을 공유할 가능성이 높아진다.

### Reversible Network

ResNet에서 나오는 Residual Connection Calculation에서 각 연산 과정을 저장할 필요없이 그냥 역산을 통해서 입력값을 알아낼 수 있다는 것을 활용하여 메모리 사용량을 줄일 수 있었따.

### LSH Attention

본 논문에서는 Query와 Key 값이 같다는 가설을 세우고 진행한다.

1. 일렬로 된 벡터 형태의 데이터 포인트에 LSH를 적용하고 같은 Hash 값을 가진 데이터 포인트끼리 버킷으로 묶는다.
2. 각 버킷에는 높은 확률로 데이터 포인트들이 불균형하게 배당될 것이다. 이 데이터 포인트를 고정된 크기의 구역으로 분절한다.
3. Attention Weigth를 계산한다.
    - 두 데이터 포인트가 같은 버킷에 있거나 같은 구역에 있거나 Attention의 도착점 데이터 포인트는 시작 데이터 포인트가 있는 구역 바로 앞 구역에 있어야 한다.
