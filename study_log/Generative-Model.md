# Generative Model

## Index

- [Generative Model](#generative-model)
  - [Index](#index)
  - [Generative Model Definition](#generative-model-definition)
  - [Generative Adversarial Nets](#generative-adversarial-nets)
    - [학습과정](#학습과정)
    - [수식](#수식)
    - [한계점](#한계점)
    - [더 공부할 것](#더-공부할-것)
    - [IS (Inception Score)](#is-inception-score)
    - [FID (Frechet Inception Distance)](#fid-frechet-inception-distance)
  - [Conditional GAN](#conditional-gan)
  - [Variational AutoEncoder](#variational-autoencoder)
    - [Autoencoder](#autoencoder)
    - [VAE](#vae)
    - [사용사례](#사용사례)

---

## Generative Model Definition

-   A statistical model of the joint probability distribution
-   An architecture to generate new data instances

생성 모델은 실존하지 않지만 있을 법한 이미지를 훈련 데이터셋의 확률 분포로 만들어낸다.

## Generative Adversarial Nets

GAN은 Generative Adversial Network의 약자로 Discriminator와 Generator 신경망이 서로 적대적으로 학습시키며 실제 데이터와 비슷한 데이터를 생성 해내는 모델이다. 이렇게 생성된 데이터에는 정해진 label값이 없기 때문에 비지도 학습 기반 생성모델로 분류된다.

GAN은 Generator와 Discriminator라는 서로 다른 2개의 네트워크로 이루어져 있으며 이 두 네트워크를 적대적으로 학습시키며 목적을 달성한다. 생성모델의 목적은 진짜 분포에 가까운 가짜분포를 생성하는 것이고 판별모델의 목적은 표본이 가짜분포에 속하는지 진짜분포에 속하는지를 결정하는 것이다. GAN의 궁극적인 목표는 "실제 데이터의 분포"에 가까운 데이터를 생성하는 것이여서 판별기가 진짜인지 가짜인지를 한 쪽으로 판단하지 못하는 경계(1/2)에서 최적 솔루션으로 간주하게 된다.

![im](./docs/gan-structure.jpeg)

G는 데이터를 생성하여 자신이 생성한 데이터를 최대한 실제 데이터처럼 만들어서 D를 속이려고 하는 것이고, D는 최대한 정확하게 구별해 내려는 방식으로 학습이 진행되어서 두 신경망의 구조가 '적대적이다'라는 것에서 adversial이 붙게 된 것이다.

### 학습과정

-   D의 학습과정

    -   m개의 noise 샘플을 noise 분포로부터 추출하고 m개의 실측데이터를 실측데이터 분포에서 추출한다.
    -   D를 SG 값만큼 상승시켜서 Update 해준다.
    -   이 과정을 k번 반복한다. (이때 G는 학습 X)

-   G의 학습과정
    -   m개의 noise 샘플을 noise 분포로부터 추출한다.
    -   G를 SG 값만큼 하강시켜서 Update 한다.
    -   다음 epoch로 넘어간다.

<strong>문제점</strong>

학습 초반에는 G의 weight들과 bias들이 제대로 학습되어있지 않기 때문에, 실측데이터와 너무나 확연하게 다른 데이터를 생성해내게 된다. 이로인해 D는 입력된 G(z)의 값에 대하여 거의 0에 가까운 값을 출력하게 되면서 gradient 값이 너무 낮아 학습이 제대로 이루어지지 않는 현상이 발생한다.

따라서 결론적으로 D(G(z))를 maximize하는 방향으로 학습 시키는 것이 그 해결책이다.

![progress](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F9916313C5AAEB6CB1B)

### 수식

![gan_formula](https://t1.daumcdn.net/cfile/tistory/995E00345AAE7B6401)

### 한계점

GAN은 기술적으로 고해상도 이미지를 생성할 수 없다는 점과 학습이 불안정하다는 점을 한계점으로 가지고 있다. 이러한 한계점은 후속 GAN 모델이 개발되면서 해결되었다. (추후 Diffusion Model의 승리 과정 학습)

-   악용가능성, 지식 재산권 이슈 등

### 더 공부할 것

DCGAN, CycleGAN 등 추가 공부가 필요하다....

-   DCGAN CNN 레이어로 고화질의 이미지를 생성할 수 있다.
-   CGAN 이미지를 다른 이미지로 변환할 수 있다.

### IS (Inception Score)

Inception Score는 GAN의 평가에 널리 쓰이는 지표이다. 이 지표는 클래스 label과 관련하여 특징적인 속성들을 잡아내기 위해 Pre-trained 신경망을 사용한다.

샘플의 조건부 분포와 모든 샘플에서 얻은 주변분포 사이의 평균적인 KL 발산 정도를 측정하는 것이다. 이 값이 높을수록 좋은 성능을 낸다고 해석할 수 있다.

![is](https://github.com/Pseudo-Lab/Tutorial-Book/blob/master/book/pics/GAN-ch1img08.png?raw=true)

### FID (Frechet Inception Distance)

FID는 생성되는 이미지의 퀄리티 일관성을 유지하기 위해 이용되는 지표이다. 실제 데이터의 분포를 활용하지 않는 단점을 보완하여 실제 데이터와 생성된 데이터에서 얻은 feature의 평균과 공분산을 비교하는 방식이다. FID가 낮을수록 이미지의 퀄리티가 더 좋아지는데 이는 실제 이미지와 생성된 이미지의 유사도가 높아지는 것을 의미한다. 즉 쉽게 말해 FID는 생성된 샘플들의 통계와 실제 샘플들의 통계를 비교하는 것이다.

## Conditional GAN

Conditional GAN은 기존의 GAN은 훈련시킨 데이터의 분포와 유사한 이미지를 잘 생성할 수 있지만 어떤 이미지를 생성하라고 통제하지는 못한다는 단점을 해결할 수 있다. Original GAN의 Discriminator와 Generator가 training 중에 external information을 받는 것이다. external information은 label이 될 수도 있고 다른 형식이 될 수도 있다.

![conditionalgan](https://velog.velcdn.com/images%2Fwilko97%2Fpost%2F343d0fe0-0bcc-4ca8-abd3-384996d84513%2Fimage.png)

CGAN은 학습과정 중에

-   G는 training dataset의 label에 대응하는 실제적인 샘플을 만드는 과정을 학습한다.
-   D는 label이 주어진 상태에서 real, fake를 구별하는 방법을 배운다.
-   Original GAN과 거의 유사하지만 auxiliary information이 추가될 뿐이다.


## Variational AutoEncoder

### Autoencoder

![ae](https://tikz.net/janosh/autoencoder.png)

- Encoder : 고차원의 입력 데이터 -> 저차원의 표현 벡터
- Decoder : 표현 벡터 -> 원본 차원

AutoEncoder은 입력데이터와 Decoder의 출력과의 손실을 최소화하는 방향으로 훈련된다. AutoEncoder의 Encoder Decoder에서는 FC를 사용하거나 Conv와 Deconv 블록이 사용된다. 

| | Conv | Conv Transpose |
|------|---|---|
| 구조 | Encoder | Decoder |
| 사용목적 | 이미지크기 줄이기 | 이미지크기 늘리기 |
| 역할 | 특징추출 | 특징복원 |

<strong>Conv Transpose</strong>는 output의 크기를 줄여서 feature을 추출하는 convolutional layer와는 반대로 output의 크기가 input보다 커지게 하는 방법으로 Upsampling의 한 종류이다. Upsampling에는 unpooling과 transposed convolution이 있다. Upsampoling은 input 이미지를 압축된 벡터로 표현했다가 원래의 input 이미지와 동일한 크기로 되돌릴 때 사용될 수 있다. 

![t_conv](https://user-images.githubusercontent.com/50395556/81541105-9401e980-93ad-11ea-87a1-a7676fbd8314.png)

Transposed Convolution은 input의 빨간색 원소를 3x3 kernel에 곱해서 output에 대응하는 자리에 집어넣고 같은 방법으로 다른 부분도 집어넣는다. 연산을 반복하면서 겹치는 부분의 값을 모두 더해준다.

### VAE

VAE는 AE를 진화시킨 버젼이라고 볼 수 있다. AE의 Bottle Neck 부분을 진화시켜서 바로 잠재 벡터화하는 것이 아니라 인코더의 각 입력을 평균 벡터와 분산 벡터로 매핑하는 과정을 거친다. 논문의 저자들은 이 두 벡터를 Reparameterization Trick을 사용하여 Backpropagation 할 수 있도록 했다고 한다. 

![vae](https://gaussian37.github.io/assets/img/dl/concept/vae/0.png)

![vae_loss](https://velog.velcdn.com/images/tobigs1617/post/e0486aee-7f50-469f-a6ec-2f45a0d285ea/image.png)


### 사용사례

- 구글이 이미지를 VAE로 디코딩하여 이미지 용량을 줄여 밴드위치를 줄였다고 한다.
- Image Segmentation 기법에 활용될 수 있다. (자율주행차)
- Noisy Image를 재구성해서 Denoise 작업을 수행할 수 있다.
- Neural Inpainting으로 이미지를 복원할 수 있다. (워터마크 없애기)