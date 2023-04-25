
### 目的
由于工作中频繁遇到很多第三方的人脸识别方案需要整合，因此考虑自己造轮子看下实现复杂度以及方便后续项目开展时做一个备选方案。


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


### 相关使用方式
face::FaceRecognizer 是一个使用 dlib 库提供人脸识别功能的 C++ 类。它允许用户在图像中检测人脸、提取面部特征、比较面部特征以及将人脸与已知个体匹配。

要使用 FaceRecognizer，用户可以创建该类的一个实例，并调用其方法来执行各种人脸识别任务。该类提供以下方法：

- GetFaceFeatureFromImg(image_path)：以图像文件路径作为输入，返回该图像中检测到的面部特征的字符串表示。

- GetSimilarityFromImg(face_feature1, image_path)：以面部特征的字符串表示和图像文件路径作为输入，返回两者之间的相似度得分。

- RecognizeImgByKnownFaces(image_path, name, break_if_first_matched)：以图像文件路径作为输入，将检测到的面部特征与已知面孔的缓存进行比较，并返回最佳匹配的名称。如果 break_if_first_matched 设置为 true，则函数在找到第一个匹配项后将停止搜索。

- AddKnownFaceFeatures(known_face_name, face_features)：将面部特征的字符串表示添加到已知面孔的缓存中，并与给定的名称相关联。

- AddKnownFaceFeatures(known_face_name, face_features_str)：与上述相同，但以字符串表示的面部特征为输入。

- EncodeFaceFeature(face_features)：以面部特征的向量作为输入，返回这些特征的二进制字符串表示。

- DecodeFaceFeature(face_features_str)：以二进制字符串表示的面部特征作为输入，返回这些特征的向量。

该类还提供了一个静态变量 matched_max_threshold，它是两个面孔被认为不匹配的相似度阈值。该变量的默认值为 0.4

要使用该类，用户可以创建一个该类的实例，并根据需要调用其方法。他们可以使用 AddKnownFaceFeatures 将已知面孔添加到已知面孔的缓存中，然后使用 RecognizeImgByKnownFaces 将新的面孔与缓存中的面孔做匹配。他们还可以使用 GetSimilarityFromImg 和 IsMatched 直接比较和匹配面孔。

