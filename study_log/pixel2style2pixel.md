# Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation

![pSp](https://eladrich.github.io/pixel2style2pixel/images/teaser.png)

This is simple paper review of pSp network short for pixel2style2pixel. It can be lack of professional concepts of deep learning because I'm a student.. Please read it with consideration.

## TL;DR

- A novel StyleGAN encoder able to directly encode real images into the $W+$ latent domain.
- A new methodology for utilizing a pre-trained StyleGAN generator to solve image-to-image translation.

## Background

As StyleGAN network can generate high quality image, many researchers tried controlling StyleGAN's latent space and performing meaningful manipulations in $W$ space. These methods follow an "invert first, edit later", where one first inverts and image into StyleGAN's latent space and the edits the latent code. This method takes long time to optimize latent vector for just one image. And pSp network solves the problem by encoding an arbitrary image directly into $W+$ space. 


## The pSp Framework

<img width="60%" src="https://production-media.paperswithcode.com/methods/ccbdc679-5dba-4457-a621-7f78e2674a22.png" />

The overall architecture of pSp network looks like this. The pSp network builds upon the representative power of a pretrained StyleGAN generator and the $W+$ latent pace. The pSp network uses encoder based on **Feature Pyramid Network**. 

<img width="50%" src="https://production-media.paperswithcode.com/methods/new_teaser_TMZlD2J.jpg" />

> FPN: This network comes from Object Detection task. It can extract semantic features more efficiently using Residual Calculate Information. [More Information](https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)


If using single 512 dimensional vector obtained from the last layer of the encoder network, such an architecture presents a strong bottleneck making it difficult to fully represent the finer details of the original image and is therefore limited in reconstruction quality.


<img width="50%" src="https://images.squarespace-cdn.com/content/v1/6292e4c9c6759c50bca53dbd/8fa634ec-0101-46c3-89ab-b67707c53f89/stylegan2.jpg" />

In StyleGAN, the different style inputs correspond to different levels of detail, which are divided into three groups, Coarse Styles, Medium Styles, Fine Styles. So, they extend an encoder backbone with a feature pyramid. And mapping network, map2style, generates 512 vector fed directly into StyleGAN generator.


## Loss Functions

<img width="50%" src="https://velog.velcdn.com/images/aioptlab/post/1867be68-3739-4181-bc15-7bef63232be3/image.png">

- E( ) represents **encoder**. And G( ) represents **StyleGAN's Generator**.
- $\hat{w}$ is the average style vector of the pretrained generator.
- Encoder aims to learn the latent code with respect to the average style vector. 


<img width="50%" src="https://velog.velcdn.com/images/aioptlab/post/8c464d41-5ca9-4651-87c3-4c6e4b9ec216/image.png" />

First, **pixel-wise $L_2$ loss**.


<img width="50%" src="https://velog.velcdn.com/images/aioptlab/post/5bae5c2c-30f9-43aa-befc-8ba29c9e915a/image.png" />

Second, using $L_{LPIPS}$ to learn perceptual similarities. It has been shown to better preserve image quality.

<img width="50%" src="https://velog.velcdn.com/images/aioptlab/post/0fb832f6-b4b1-43e1-8c24-7d6d80bf861c/image.png" />

Thirds, using **Regularization Loss** for output latent style vectors closer to average latent vector.  

<img width="50%" src="https://velog.velcdn.com/images/aioptlab/post/4ccbaf74-a9f6-46a3-87f8-6084b7f69f1f/image.png" />

Finally, to preserve face image identity, they use pretrained ArcFace network. And they calculate cosine similarity loss meausing the cosine between the output image and its suorce.


## The Benefits of the StyleGAN Domain

- As they makes their model operate *globally* instead of *locally*, pSp is different from many standard image-to-image translation model.
- They suport *multi-modal synthesis* because they use pretrained StyleGAN Generator. 


**Thank you for reading!**





