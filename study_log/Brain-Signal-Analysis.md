# 뇌에 대한 공부 기록

<!-- TOC -->

- [뇌에 대한 공부 기록](#%EB%87%8C%EC%97%90-%EB%8C%80%ED%95%9C-%EA%B3%B5%EB%B6%80-%EA%B8%B0%EB%A1%9D)
- [뇌파 측정과 동적신경영상 이론 및 응용](#%EB%87%8C%ED%8C%8C-%EC%B8%A1%EC%A0%95%EA%B3%BC-%EB%8F%99%EC%A0%81%EC%8B%A0%EA%B2%BD%EC%98%81%EC%83%81-%EC%9D%B4%EB%A1%A0-%EB%B0%8F-%EC%9D%91%EC%9A%A9)
    - [뇌파 기슬 개론 및 역사](#%EB%87%8C%ED%8C%8C-%EA%B8%B0%EC%8A%AC-%EA%B0%9C%EB%A1%A0-%EB%B0%8F-%EC%97%AD%EC%82%AC)
        - [Bioelectromagnetics 생체전자기학](#bioelectromagnetics-%EC%83%9D%EC%B2%B4%EC%A0%84%EC%9E%90%EA%B8%B0%ED%95%99)
        - [뇌전도 vs 뇌자도](#%EB%87%8C%EC%A0%84%EB%8F%84-vs-%EB%87%8C%EC%9E%90%EB%8F%84)
    - [뇌파의 발생 원리](#%EB%87%8C%ED%8C%8C%EC%9D%98-%EB%B0%9C%EC%83%9D-%EC%9B%90%EB%A6%AC)
        - [안정 막전위](#%EC%95%88%EC%A0%95-%EB%A7%89%EC%A0%84%EC%9C%84)
        - [활동 전위](#%ED%99%9C%EB%8F%99-%EC%A0%84%EC%9C%84)
        - [Structure of Human Head](#structure-of-human-head)
    - [뇌파의 측정 기술](#%EB%87%8C%ED%8C%8C%EC%9D%98-%EC%B8%A1%EC%A0%95-%EA%B8%B0%EC%88%A0)
        - [뇌파의 분류](#%EB%87%8C%ED%8C%8C%EC%9D%98-%EB%B6%84%EB%A5%98)
        - [응용 분야](#%EC%9D%91%EC%9A%A9-%EB%B6%84%EC%95%BC)
- [합성곱 신경망을 이용한 뇌파 해석 기법](#%ED%95%A9%EC%84%B1%EA%B3%B1-%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EB%87%8C%ED%8C%8C-%ED%95%B4%EC%84%9D-%EA%B8%B0%EB%B2%95)

<!-- /TOC -->

# 뇌파 측정과 동적신경영상 이론 및 응용 

> 연세대학교 의료공학연구센터 단기교육강좌 강의자료 참고

## 뇌파 기슬 개론 및 역사

### Bioelectromagnetics 생체전자기학

생체에 대한 전자기 현상을 대상으로 한 의학이다.

- Measurement
    - Brain Activity -> EEG, MEG
    - Cardiac Activity -> ECG, MCG
    - Gastric Activity -> EGG, MGG
    - Bioimpedence -> EIT, MIT
- Stimulation
    - Brain Stimulation
    - etc

### 뇌전도 vs 뇌자도

**EEG나 MEG는 비침습적이면서 뛰어난 시간 분해능을 가진다.**

> 시간분해능: 물체의 움직임을 잘게 쪼개서 관찰할 수 있는가
>
> 공간분해능: 정지된 화면을 정밀하게 관찰할 수 있는가

![우와ㅏ](https://www.researchgate.net/publication/278826818/figure/fig3/AS:614336141746177@1523480558607/Spatial-and-temporal-resolutions-of-diierent-modalities-commonly-used-for-functional.png)

- 뇌전도(EEG): 뇌 표면에서 발생하는 전기 퍼텐셜의 차이를 전극을 이용하여 측정
- 뇌자도(MEG): 뇌로부터 발생되는 자장신호를 SQUID 센서를 이용하여 측정

**뇌전도**

EEG는 시간 분해 신호이므로 목적과 상관없는 일시적 드래프트가 있는 경우가 매우 잦다. 

- 측정 방법
    - 머리 표면에 전극을 부착하여 머리 표면의 전위차를 측정한다. 시스템 가격이 저렴 설치 간단
    - 뇌는 자기적으로 투명하지만 전기적으로는 투명하지 않다. => 측정 신호가 왜곡된다. => MEG보다 공간 분해능이 떨어지는 것으로 인식
- 신호처리 => 잡음 처리 필요!
    - 하드웨어적 신호처리 (A/D 변환, 증폭) 
    - 소프트웨어적 신호처리 (Baseline correction, Filtering, Artifact rejection)
        - Baseline Correction: 기준선과 자극 후 간격의 모든 시점에서 기준선 기간의 평균을 빼는 것


**뇌자도**

생체 자기 신호는 매우 미약하고 환경 자기 잡음에 매우 민감하기 때문에 고감도의 자장 측정 기술이 필요하다. 


## 뇌파의 발생 원리

### 안정 막전위

안정시 세포 내부가 세포 외부에 비하여 6-90mV의 음전하를 띠고 있다. 외부 자극이 없는 상태에서 세포 안 밖에 존재하는 전위차를 **안정막전위**라고 한다. 

안정상태에서는

- Extracellular medium: Cl- or Na+
- Intracellular medium: A- or K+

각 이온들은 지방을 통해서 이동하지 못하고 이온 통로를 통해서 이동하는 *선택적 투과성*을 가진다.

### 활동 전위

> 활동 전위: 세포막 전위의 변동

1. 탈분극: 문턱치 이상의 자극이 가해지면 헉 => Na+ 통로 개방 => Na+가 세포 안으로 드루와 => 막전위 감소 => 세포 안팎의 전위가 역전
2. 재분극: Na+ 닫히고 K+ 열려서 세포 밖으로 K+ 이동 => 막전위 복원
3. 과분극: K+ 너무 많이 나가서 안정막전위보다 더 낮게 음전하로 기운다
4. 안정막전위: 일정 시간 이후 다시 복원

만약 외부에서 어느 자극이 들어왔다!! 그 자극이 문턱치 이상의 자극이면 이 자극을 뉴런들에게 전파한다. 마치 NN의 Forward Pass와 같다. 

### Structure of Human Head

- Scalp: 머리카락
- Skull: 두개골
- Cerebrospinal fluid: 뇌척수액 
    - 뇌, 척수 등을 채우고 있는 액체로 기계적 충격등을 담당한다.
- Cerebral Cortex: 대뇌피질
    - 대뇌피질은 뇌에서 가장 큰 부분을 차지하고 가장 중요한 부분이다. 우리가 사고하는 등의 활동을 하는 것은 모두 대뇌피질이 있기에 가능한 것이다. 많은 영역으로 나뉘어지며 측두엽, 전두엽 등이 있다.

![graywhite](https://cdn.technologynetworks.com/tn/images/body/brainimage11566227404332.png)

- Gray Matter: 회색질
    - 사고, 기억 등의 고차원적인 뇌 기능을 담당하는 곳으로 뉴런의 체와 가지돌기가 밀집되어 있다.
- White Matter: 백질
    - 회색질의 가지돌기를 서로 연결해주는 신경섬유들이 위치한다. 

> To know the structures of cerebral cortex, especially along cortical surface, is very important for neuroelectromagnetic inverse problem.

=> 우리는 뉴런의 신호 전달 과정에서 시냅스에서 발생하는 후시냅스 전위가 전자기장을 생성하는 데 큰 역할을 한다고 본다. 한 번의 많은 활동 전위가 발생하는 것은 어렵기 때문이다. 그러나 여러 수상돌기가 평행적으로 배열되어 있다면 측정 가능한 전자기장을 유도할 수 있다. 

## 뇌파의 측정 기술

- Baseline Correction
    - Baseline: 측정 도중 발생하는 shift 현상 (as Noise, Brain Activity, Muscle Tension)
    - 1. 자극을 주기전의 파형을 baseline interval이라고 하고 이 동안의 신호를 채널 별로 평균을 계산하여 빼준다.
    - 2. 전체 신호에 걸쳐 나타나는 선형적인 trend를 제거
- Filtering
    - 일반적으로 Bandpass Filter 사용
- Artifact Removal 
- Segmentation and Averaging
    - Segmentation: 자극 또는 반응에 대해서 일정 구간으로 나누는 작업
    - Averaging: 특정한 자극의 type에 대해서 평균을 내는 작업


### 뇌파의 분류

- 델타파: 정상인의 깊은 수면상태나 신생아에서 주로 나타난다. 그리고 델파 상태에서는 많은 양의 성장 호르몬을 발생시킨다.
- 쎄타파: 정서적으로 안정된 상태나 수면에 빠지기 전에 나타난다. 지각과 꿈의 경계상태라고도 부른다. 예기치 않은 꿈과 같은 마음의 이미지를 종종 동반하게 되고 그 이미지는 생생한 기억으로 이어지는 경험을 하게 된다. 이것은 곧 갑작스러운 통찰력 또는 창조적 아이디어로 연결되기도 하고 초능력이라는 비현실적이고 미스터리한 환상적 상태로 비춰지기도 한다. 
- 알파파: 명상 같은 편안한 상태에서 나타나며 스트레스 해소 및 집중력 향상에 도움을 준다. 
- 베타파: 긴장, 흥분 상태 등 활동할 때에 나타난다. 운동력 향상에 도움을 주고, 의식이 깨어 있을 때의 뇌파이다. 
- 감마파: 주로 흥분했을 때 나타난다. 

### 응용 분야

1. Neurofeedback
    비침습적인 방법으로 특정 뇌파 또는 뇌 활성화를 유도하는 Neuromodulation의 일종이다.
2. EEG/MEG 기반 BCI
    상상을 통하여 기계 또는 컴퓨터를 제어하는 기술
3. 감성공학, 수면감시 - Rhythmic Activity
4. 뇌질활 진단


# 합성곱 신경망을 이용한 뇌파 해석 기법

뇌-컴퓨터 인터페이스는 뇌로부터 발생하는 전기적인 신호로 임의의 기계를 제어할 수 있는 기술이다. 

뇌파를 얻는 실험으로부터 충분한 양의 데이터를 확보하기에는 어려울 수 있어서 본 연구에서는 뇌파 데이터의 증대를 위해서 랜덤 필드를 활용한다. 이 랜덤 필드를 기계학습에 활용하여 뇌파 분류 알고리즘의 성능을 향상시키는 방안을 제시한 의의가 있다.

- 생성방법
    - 시간 영역의 뇌파 데이터를 푸리에 변환을 적용해 주파수 영역의 데이터로 변환한다.
    - 가우시안 랜덤 필드에서 데이터를 샘플링한다.
    - 생성된 데이터들의 공분산 행렬을 기존 데이터의 공분산 행렬과 하삼각행렬을 도입하여 생성 데이터와 기존 데이터의 공분산 행렬을 같게 한다.
    - 새로운 하삼각행렬의 원소를 구한 후 기존의 Cholesky-decomposition 방법을 이용한 가우시안 랜덤 필드를 마찬가지로 적용하여 새로운 데이터를 생성한다.

# EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces

기존의 EEG Signal을 활용한 감정 분석 같은 Task에서는 Domain Specific하도록 EEG 데이터 중에 필요한 일부를 선택적으로 전처리하여 사용했다. 본 연구에서는 CNN 아키텍처를 활용하여 EEG 데이터를 효과적으로 분석할 수 있는 아키텍처를 제안하고 훈련된 모델을 활용하여 다양한 Task에 쓰일 수 있음을 보여서 일반화 될 수 있음을 보였다. 

## Introduction

일반적인 BCI 분석 과정

1. Data Collection Stage
    뇌파 데이터를 수집하는 과정이다.

2. Signal Processing Stage
    수집된 데이터를 전처리하고 정리하는 과정이다. 

3. Feature Extraction Stage
    Task를 수행하기 위한 의미있는 뇌파 데이터를 추출하는 과정이다.

4. Classification Stage
    데이터로부터 예측하는 과정이다.

5. Feedback Stage
    피드벡을 통해서 수정하는 과정이다. 

이와 같은 과정을 통해서 분석을 하기 위해서는 많은 선행 지식이 필요하고 의미있는 뇌파 데이터만을 사용하면서 잠재적으로 관련이 있을 수 있는 EEG 특징이 제외될 수 있는 문제가 있다. 본 논문에서는 이런 문제를 해결했다고 한다. 또한 EEG 특징을 시각화할 수 있는 새로운 방법을 제시하여 좋은 분류 성능을 보일 수 있도록 했다고 한다. 

## Materials and Methods

### Data Description

- Event-Related Potential (ERP) BCIs: 시간 정보 
    특정 시간 동안 발생한 이벤트에 대한 뇌파 신호를 분석한다. 다시 말해서 ERP는 시간 자극에 대한 반응 측정하고 분석한다. 

![ERP](https://www.brainlatam.com/uploads/tinymce/EEG%20ERP%20Analysis%20of%20Event-Related%20Potentials-1644866350.webp)


- Event-Related Spectrum Perturbation (ERSP) BCIs: 주파수 정보 
    이벤트와 관련된 주파수 변화를 분석한다. 

![ERSP](https://eeglab.org/assets/images/newtimefplot1.png)


### Classification Methods