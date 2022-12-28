# 수학 개념과 용어 정리..

## 선형 대수학

## 통계학

### 확률분포

확률 변수가 특정한 값을 가질 확률을 나타내는 함수를 의미한다.

### 이산확률분포

확률변수 X의 개수를 정확히 셀 수 있을 때 이산확률분포라고 말한다.

### 연속확률분포

확률변수 X의 개수를 정확히 셀 수 없을 때 연속 확률분포라고 말한다.

ex) 정규분포

<strong>정규 분포(Gaussian Distribution)</strong>

연속 확률 분포의 하나이다. 정규분포는 수집된 자료의 분포를 근사하는 데에 자주 사용된다. 이것은 중심극한정리에 의하여 독립적인 확률변수들의 평균은 정규분포에 가까워지는 성질이 있기 때문이다.

사람의 얼굴에도 통계적인 평균치가 존재할 수 있다. 모델은 이를 수치적으로 표현할 수 있게 된다. 이때의 확률 분포는 이미지에서의 다양한 특징들이 각각의 확률 변수가 되는 분포를 의미한다.

<strong>다변수 확률 분포</strong>

![multivariable](https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Multivariate_Gaussian.png/330px-Multivariate_Gaussian.png)

<strong>가우시안 노이즈</strong>

![gaussian](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCbW0g%2FbtqN005Ge5Y%2FkUVZpwTKpdhOETVC94CAkK%2Fimg.png)

이런 식으로 사진이 지지직 거리는 느낌의 잡음을 가우시안 노이즈라고 한다. 이름이 가우시안 노이즈인 이유는, 이름처럼 가우스 함수에 따른 분포를 따르고 있기 때문이다.

```python
def make_noise(std, gray):
    height, width = gray.shape
    img_noise = np.zeros((height, width), dtype=np.float)
    for i in range(height):
        for a in range(width):
            make_noise = np.random.normal()  # 랜덤함수를 이용하여 노이즈 적용
            set_noise = std * make_noise
            img_noise[i][a] = gray[i][a] + set_noise
    return img_noise
```
