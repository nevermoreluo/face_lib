//
// Created by SG220 on 2023/4/11.
//

#include <gtest/gtest.h>
#include "face.h"

TEST(FaceCheck, BasicAssertions) {
    // base check
    EXPECT_STREQ("hello", "hello");

    face::FaceRecognizer fr;

    std::string base_face_name = "Jay";
    std::string base_image_path = "test_faces/face_1_1.jpg";

    std::string f_image_path_1_2 = "test_faces/face_1_2.jpg";

    std::string f2_name = "ym";
    std::string f_image_path_2_1 = "test_faces/face_2_1.jpg";
    std::string f_image_path_2_2 = "test_faces/face_2_2.jpg";
    auto base_feature = fr.GetFaceFeatureFromImg(base_image_path);

    auto base_face_code = face::FaceRecognizer::EncodeFaceFeature(base_feature);
    auto f_1_2_sim = fr.GetSimilarityFromImg(base_face_code, f_image_path_1_2);


    EXPECT_LT(f_1_2_sim, face::FaceRecognizer::matched_max_threshold);
    EXPECT_EQ(true, face::FaceRecognizer::IsMatched(f_1_2_sim));

    EXPECT_NE(true, face::FaceRecognizer::IsMatched(fr.GetSimilarityFromImg(base_face_code, f_image_path_2_1)));
    EXPECT_NE(true, face::FaceRecognizer::IsMatched(fr.GetSimilarityFromImg(base_face_code, f_image_path_2_2)));

    /// ------------------------------------------------
    /// Add to face to known cache, detect face
    /// GetFaceFeatureFromImg spent too many time
    fr.AddKnownFaceFeatures(base_face_name, base_feature);

    std::string face_name;
    EXPECT_EQ(0, fr.RecognizeImgByKnownFaces(f_image_path_1_2, face_name));
    EXPECT_EQ(base_face_name, face_name);

    //std::string face_name_1_3;
    //EXPECT_EQ(0, fr.RecognizeImgByKnownFaces(f_image_path_1_3, face_name_1_3));
    //EXPECT_EQ(base_face_name, face_name_1_3);
    
    
}



