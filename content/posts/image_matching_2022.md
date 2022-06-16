---
title: "Models and Upscaling — Image Matching Challenge 2022"
date: "2022-06-12T23:46:37.121Z"
template: "post"
draft: false
slug: "imagematching2022"
category: "Competitions"
tags:
  - "Computer Vision"
  - "Image Matching"
  - "Stereo Matching"
  - "3D"
description: "In this article I'm going to discuss the field on stereo matching and make a recap of solution our team made for the Image Matching Challenge 2022
from Google Research"
socialImage: "/media/paper_review_swin_transformer/review_collage.png"
---

I'm excited to say that our team (Kenjiro, Ohta and me) finished **17th** in Image Matching Challenge 2022 from Google Research. The topic of stereo matching and, 
generally, 3D computer vision was new to me, but I managed to try some great methods I'd like to share with you.

![collage](/media/image_matching.png)

## Overview

This part was taken from the [page](https://www.kaggle.com/competitions/image-matching-challenge-2022/overview/problem-definition) of problem definition from Image Matching Challenge. 

### Problem Definition

The goal is to predict the relative pose between two images.

Important concepts:

- *Calibration matrix* **K** captures camera properties that determine the transformation between 3D points and 2D coordinates (camera intrinsics).

- *Rotation matrix* **R** and *translation matrix* **T** capture the 6-degree-of-freedom pose (position and orientation) of the camera in a global reference frame (camera extrinsics).

- *Fundamental matrix* **F** encapsulates the projective geometry between two views of the same scene. Content of the scence does not affect the result. Only intristics and extristics matter.

### Projective Geometry

Projective geometry in computer vision deals with transformations between 3D (world) and 2D (image) perspectives. Example with the *pinhole camera*.

| ![pinhole camera](https://kornia.readthedocs.io/en/latest/_images/pinhole_model.png) |
|:--:| 
| *Pinhole camera. Image belongs to [Kornia](https://kornia.readthedocs.io/en/latest/geometry.camera.pinhole.html).* |

The projection of a 3D point **M(i)**, in meters, into 2D coordinates **m(i)**, in pixels, can be simpy written as **m(i) = KM(i)**, 
where **K** is the calibration of insintrics matrix:

<img src='https://latex.codecogs.com/svg.image?K=\begin{pmatrix}f(x)&space;&&space;0&space;&&space;u(0)&space;\\0&space;&&space;f(y)&space;&&space;v(0)&space;\\0&space;&&space;0&space;&&space;1&space;\\\end{pmatrix}' alt='Pinhole camera' width="200" style="background-color:white;"/>

Where **f(x), f(y)** are the *focal lengths* on **x** and **y**, and **u(0), v(0)** are *principal point offsets*. The images have no radial distortion.

However, that equation is only valid for 3D points in camera's coordinate system. Given points in the "world" coordinate system (common across the different cameras in the scene), and a camera at the position encoded by **3 x 3** rotation matrix **R** and a 3-dimensional translation vector **T**, the projection can be written as:

**m(i) = K(RM(i) + T)**

### Epipolar Geometry

Epipolar geometry is the projective geometry between two views. It depends only on extrinsics and intrinsics. In computer vision, this relationship is tipically expressed as the fundamental matrix, a **3 x 3** matrix of rand 2. Given a 3D point **M(i)**, in meters, and its projections on two different cameras **m(i)** and **m'(i)**, in pixels, the fundamental matrix for these two views **F** must satisfy:

<img src="https://latex.codecogs.com/svg.image?{m_{i}'}^{T}Fm_{i}&space;=&space;0" alt='Pinhole camera' width="200" style="background-color:white;"/>

## Approaching Image Matching

Classical stereo matching pipeline consists of:

1. Feature extraction (handcraftred feature, deep learning descriptors or end-to-end pipelines).
2. Feature matching by nearest-neighbor search in descriptor space.
3. Outlier pre-filtering (optional)
4. Outlier filtering with RANSAC

### Feature Extraction

There are many ways to extract features from an image. At first, you can use handcrafted descriptors like SIFT, ORB or AKAZE. They are clearly 
outperformed by modern deep learning methods, but at the same time they are great as a starting point.

SIFT is a timeless classic invented by David Lowe in 1999. Briefly, it finds points of interest in an image and computes Difference of Gaussians (DoG) to finally determine desciptors.

| ![sift](https://docs.opencv.org/4.x/sift_dog.jpg) |
|:--:| 
| *SIFT. Image belongs to OpenCV.* |

Then you can try more recent methods which consists of descriptors learned on DoG keypoints like L2-Net, Hardnet, Geodesc or SOS-Net. Also
there are some end-to-end solutions like Superpoint or R2D2.

### Outlier Pre-Filtering

Outlier pre-filtering is an optional step in which you can use neural networks-based models to remove some outliers before feeding matches into RANSAC.
We tried a combination of DISK + OANet in our solution, where OANet was used for additional outlier detection. Unfortunately, we did not succeed with
this approach.

### RANSAC

Main method for outlier filtering is RANSAC. It runs iteratively over a set of keypoints. For futher details refer to [wiki page](https://en.wikipedia.org/wiki/Random_sample_consensus).

## Our solution

Main idea of our solution is about using multiple models by concatenating acquired keypoints and then run RANSAC on top of them. 

### Modeling

Our solution includes:

1. **LoFTR DS and OT**. LoFTR is SOTA-level transformer-based model for image matching proposed in this [[paper](https://arxiv.org/pdf/2104.00680.pdf)], check it out. 
There are two variations Dual-Softmax and Optimal Transport (DS and OT). Our team used default settings for DS model and 0.55 match_coarse threshold for OT.
2. **SuperPoint/SuperGlue**. That's a classical combination which has an amazing performance [[paper](https://arxiv.org/pdf/1911.11763.pdf)]. 
3. **DKM**. Geometric matching approach [[paper](https://arxiv.org/pdf/2202.00667.pdf)]
4. **MatchFormer-LargeLA**. Another SOTA transformer-based model. Refer to [[paper]](https://arxiv.org/pdf/2203.09645.pdf). Similarly to LoFTR for this
model we used 0.55 match_coarse threshold.

### Upscaling with Lanczos interpolation over 8×8 pixel neighborhood

Part which had the biggest impact on our solution was **upscaling** with Lanczos interpolation. Generally speaking, we upscaled images for all of our models except MatchFormer by a factor of **1.5x**, and then made them divisible by 8. In the end, position of keypoints was adjusted using the same scale factor. We think that CNNs used in many models, especially in LoFTR, can simply extract richer local features from upscaled images.

~~~~python
img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LANCZOS4)

w, h = img.shape[1], img.shape[0]
w, h = w // 8 * 8, h // 8 * 8

img = img[:h, :w, :]
~~~~

### Outlier detection

For outlier detection we incorporated simple MAGSAC from OpenCV:

~~~~python
cv2.findFundamentalMat(keypoints_1, keypoints_2, cv2.USAC_MAGSAC, ransacReprojThreshold=0.25, confidence=0.99999, maxIters=100000)
~~~~

### What did not work for us?
Here's a list of things which unfortunately didn't improve our score:

* SuperGlue re-training or fine-tuning (we did this to combine it with DISK descriptors)
* DISK + OANet
* Calculating additional keypoints from rotated images
* Inverse image order for LoFTR

## Example

A small example of how our solution perform:

![keypoints](/media/keypoints.png)
