
#include <dlib/opencv.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_io.h>
#include <dlib/serialize.h>

#include <opencv2/opencv.hpp>

#include "face.h"

using namespace face;

float FaceRecognizer::matched_max_threshold = 0.4;

face_features_t FaceRecognizer::GetFaceFeatureFromImg(const std::string& image_path) {
    face_features_t result;

    // Load the image using OpenCV
    cv::Mat img = cv::imread(image_path);

    // Initialize the face detector
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    // Convert the image to dlib format
    dlib::cv_image<dlib::rgb_pixel> dlib_img(img);

    // Detect faces in the image
    std::vector<dlib::rectangle> faces = detector(dlib_img);
    
    
    if (faces.empty()) return result;

    auto sp = GetLandmarkDetector();
    auto net = GetFaceRecognizer();
    
    // Extract face features for each detected face
    // std::vector<dlib::matrix<float, 0, 1>> face_features;
    for (auto& face : faces) {
        // Get the facial landmarks
        dlib::full_object_detection landmarks = sp(dlib_img, face);

        // Extract the face features
        dlib::matrix<dlib::rgb_pixel> face_chip;
        dlib::extract_image_chip(dlib_img, dlib::get_face_chip_details(landmarks, 150, 0.25), face_chip);
        /*face_features.push_back(net(face_chip));*/
        result = net(face_chip);
        break;
    }

    return result;
}

std::vector<face_features_t> FaceRecognizer::GetQueryFaceFeaturesFromImg(const std::string& image_path)
{
    // Initialize the face detector
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    std::vector<dlib::matrix<float, 0, 1>> query_face_features;
    try {
        // Extract face features for the query image
        cv::Mat query_img = cv::imread(image_path);
        if (query_img.empty()) {
            // Failed to read the image, handle the error
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return query_face_features;
        }

        dlib::cv_image<dlib::rgb_pixel> query_dlib_img(query_img);
        std::vector<dlib::rectangle> query_faces = detector(query_dlib_img);

        for (auto& face : query_faces) {
            dlib::full_object_detection landmarks = GetLandmarkDetector()(query_dlib_img, face);
            dlib::matrix<dlib::rgb_pixel> face_chip;
            dlib::extract_image_chip(query_dlib_img, dlib::get_face_chip_details(landmarks, 150, 0.25), face_chip);
            query_face_features.push_back(GetFaceRecognizer()(face_chip));
        }
    }
    catch (std::exception& e) {
        // Handle the exception
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return query_face_features;
    }
    return query_face_features;
}

float FaceRecognizer::GetSimilarity(const std::string& face_feature_str, std::vector<face_features_t> query_face_features)
{
    float min_distance = std::numeric_limits<float>::max();
    if (face_feature_str.empty()) return min_distance;
    auto face_feature = DecodeFaceFeature(face_feature_str);
    if (face_feature.size() == 0) return min_distance;
    return GetSimilarity(face_feature, query_face_features);
}


float FaceRecognizer::GetSimilarity(face_features_t& face_feature, std::vector<face_features_t> query_face_features)
{
    float min_distance = std::numeric_limits<float>::max();
    // Initialize the face detector
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    if (query_face_features.empty()) return min_distance;
    
    // Compare the query face features with saved face features
    for (auto& query_face_feature : query_face_features) {
        float distance = dlib::length(query_face_feature - face_feature);
        if (distance < min_distance) {
            min_distance = distance;
        }
    }
    return min_distance;
}


bool FaceRecognizer::Match(const std::string& face_feature_str, std::vector<face_features_t> query_face_features)
{
    if (face_feature_str.empty()) return false;
    auto face_feature = DecodeFaceFeature(face_feature_str);
    if (face_feature.size() == 0) return false;
    return Match(face_feature, query_face_features);
}

bool FaceRecognizer::Match(face_features_t& face_feature, std::vector<face_features_t> query_face_features)
{
    return IsMatched(GetSimilarity(face_feature, query_face_features));
}

float FaceRecognizer::GetSimilarityFromImg(const std::string& face_feature_str, const std::string& image_path)
{
    float min_distance = std::numeric_limits<float>::max();
    if (face_feature_str.empty()) return min_distance;
    auto face_feature = DecodeFaceFeature(face_feature_str);
    if (face_feature.size() == 0) return min_distance;

    std::vector<dlib::matrix<float, 0, 1>> query_face_features = GetQueryFaceFeaturesFromImg(image_path);

    return GetSimilarity(face_feature, query_face_features);
}


int FaceRecognizer::RecognizeImgByKnownFaces(const std::string& image_path, std::string& name, bool break_if_first_matched)
{
    if (known_face_features_cache.empty()) {
        std::cerr << "Empty known faces when call " << __FUNCTION__ << std::endl;
        return -3;
    }

    // Initialize the face detector
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    std::vector<dlib::matrix<float, 0, 1>> query_face_features = GetQueryFaceFeaturesFromImg(image_path);

    if (query_face_features.empty()) {
        // no faces in image
        return -2;
    }

    float min_distance = std::numeric_limits<float>::max();

    // Compare the query face features with saved face features
    for (auto& query_face_feature : query_face_features) {
        for (auto& face_pair : known_face_features_cache) {
            auto face_feature = face_pair.second;
            float distance = dlib::length(query_face_feature - face_feature);
            if (distance > min_distance || !IsMatched(distance)) {
                continue;
            }
            min_distance = distance;
            name = face_pair.first;
            if (break_if_first_matched)
                return 0;
        }
    }
    return IsMatched(min_distance) ? 0 : -1;
}


void FaceRecognizer::AddKnownFaceFeatures(std::string known_face_name, face_features_t face_features) 
{
    if (known_face_name.empty()) {
        std::cerr << "Can not add known face features with empty known_face_name " << std::endl;
        return;
    }
    known_face_features_cache.insert(std::pair<std::string, face_features_t>{ known_face_name , face_features });
}

void FaceRecognizer::AddKnownFaceFeatures(std::string known_face_name, const std::string& face_features)
{
    AddKnownFaceFeatures( known_face_name , DecodeFaceFeature(face_features));
}


// note buffer type return be careful
std::string FaceRecognizer::EncodeFaceFeature(face_features_t face_feature)
{
    // Serialize the face feature to a stringstream
    std::stringstream ss;
    dlib::serialize(face_feature, ss);

    // Convert the serialized data to a string
    std::string face_feature_str = ss.str();
    return face_feature_str;
}

face_features_t FaceRecognizer::DecodeFaceFeature(const std::string& face_features_str)
{
    // Convert the string to a stringstream
    std::stringstream ss(face_features_str);

    // Deserialize the face features from the stringstream
    dlib::matrix<float, 0, 1> loaded_face_features;
    dlib::deserialize(loaded_face_features, ss);
    return loaded_face_features;
}
