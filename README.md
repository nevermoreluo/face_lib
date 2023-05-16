
[![CodeFactor](https://www.codefactor.io/repository/github/nevermoreluo/face_lib/badge)](https://www.codefactor.io/repository/github/nevermoreluo/face_lib)
[![Build Status](https://github.com/nevermoreluo/face_lib/actions/workflows/cmake.yml/badge.svg)](https://github.com/nevermoreluo/face_lib/actions/workflows/cmake.yml)
[![Doc Status](https://github.com/nevermoreluo/face_lib/actions/workflows/doc.yml/badge.svg)](https://github.com/nevermoreluo/face_lib/actions/workflows/doc.yml)
[![LICENSE](https://img.shields.io/github/license/nevermoreluo/face_lib?style=plastic)](https://github.com/nevermoreluo/face_lib/blob/main/LICENSE)
![GitHub last commit](https://img.shields.io/github/last-commit/nevermoreluo/face_lib?style=plastic)

### 目的
由于工作中频繁遇到很多第三方的人脸识别方案需要整合，因此考虑自己造轮子看下实现复杂度以及方便后续项目开展时做一个备选方案。

### 相关使用方式
[Docs](https://nevermoreluo.github.io/face_lib/classface_1_1FaceRecognizer.html)

### 主要功能
目前主要需要实现对比两张静态图片人脸识别
实现：  
依赖opencv读取图片信息，使用dlib将图片中的人脸特征值提取出来（建议将基础对照的人脸姓名和人脸特征值另外存储），使用dlib::length对比两张人脸特征值的差值超过阈值即认为相同

使用：  
GetFaceFeatureFromImg从图片读取生成人脸特征值，用户可以选择使用EncodeFaceFeature存储以及下次加载特征值，获取特征值后对比得到相似度，推荐使用IsMatched函数对比



### TODO
阈值还需要进一步测试调整，可能`res/test_faces`中的网图精修过度了，需要一些实例图片进一步调整以下阈值，进一步测试确定  
~~其实可以进一步通过特征值位置做进一步的活体检测，张嘴眨眼等活体检测~~

### 依赖
- vcpkg
- cmake
- gtest(可在cmake中设置关闭`set(ENABLE_GTEST OFF)`)
- dlib_face_recognition_resnet_model_v1.dat 人脸识别库识别图像中人脸位置的数据集 http://dlib.net/files/
- shape_predictor_68_face_landmarks.dat  人脸识别库识别人脸68个特征位置的数据集 http://dlib.net/files/

数据集文件可以在http://dlib.net/files/下载也可以在项目的res目录内获得


[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fnevermoreluo%2Fface_lib.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fnevermoreluo%2Fface_lib?ref=badge_large)
