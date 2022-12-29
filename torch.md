# Pytorch 모르는 것들 정리

## Embedding Layer

Embedding 이라는 말은 NLP에서 많이 만날 수 있다. 이산적 범주형 변수를 sparse한 one-hot 인코딩으로 나타내는 것이 아니라 연속적인 값을 가지는 벡터로 표현하는 방법을 말한다. 즉, 수많은 단어를 one-hot 인코딩하면 수치로는 표현이 가능하겠지만 대부분의 값이 0이 되어버리기 때문에 매우 sparse 해지기 때문에 임의의 길이의 실수 벡터로 밀집되게 나타나게 하는 것을 일련의 임베딩이라고 한다. 각 카테고리를 나타내는 벡터를 임베딩 벡터라고 한다.

![embedding](./docs/%08torch-embedding.jpeg)

```python
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type)
```

-   num_embeddings: 임베딩 벡터를 생성할 전체 범주의 개수
-   embedding_dim: 임베딩 벡터의 차원
-   padding_idx: 지정된 인덱스에 대해서는 학습이 진행되지 않는다.
-   max_norm: 특정 실수가 주어지고 임베딩 벡터의 norm이 이 값보다 크면 norm이 이 값에 맞추어지도록 정규화된다.
