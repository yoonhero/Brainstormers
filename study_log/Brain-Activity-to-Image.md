# Brain Activity to Image

## ○ Brain2Image: Converting Brain Signals into Images(2017)

- 본 연구에서는 EEG 장비로 측정한 뇌의 시각 자극의 디코딩 작업이 GAN과 VAE와 같은 생성형 인공지능 모델을 통해서 성공적으로 수행될 수 있음을 보여줬다. LSTM을 쌓아올린 구조의 인코더를 사용하여 사람이 상상하고 있는 것을 EEG 뇌파 데이터를 인코딩한 잠재 벡터에 대해서 분류만 하던 기존의 연구와 다르게 사람이 상상하는 것을 GAN, VAE와 같은 생성형 인공지능 모델을 활용하여 그려낼 수 있는 것을 보여준 의의가 있다. 결과적으로 EEG로 수집한 뇌파 데이터로 의미론적인 이미지를 효과적으로 생성할 수 있었다고 한다.

## ○ Hyperrealistic neural decoding for reconstructing faces from fMRI activations via the GAN latent space(2022)

- 본 연구에서는 fMRI 뇌 활동 데이터를 인코딩하여 StyleGAN의 잠재 벡터로 사용하여 초고화질 사람 얼굴을 재구성하는 Task를 성공적으로 수행할 수 있음을 보여줬다. 기존의 연구 방법과는 다르게 사전 학습된 StyleGAN 네트워크를 활용하여 인코더만 학습시켜서 좋은 품질의 데이터를 생성할 수 있었다고 한다. 

> We show the the latent vectors used for generation effectively capture the same defining stimulus properties as the fMRI measurements.

*Neural Decoding*: brain responses -> Feature Map -> sensory stimuli

- Brain responses -> Feature Map = *Linear*
- Feature Map -> sensory stimuli = *Nonlinear*

Why Feature Map?: data efficient (given that neural data is scarce) & test alternative hypothesis