#pragma once

#include <string>

#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <algorithm>

#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <vector>
#include <mutex>


namespace face
{
    using face_features_t = dlib::matrix<float, 0, 1>;
    using face_features_map_t = std::unordered_map<std::string, face_features_t>;


    using namespace std;


    // ----------------------------------------------------------------------------------------

    template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;
    template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;
    template <int N, template <typename> class BN, int stride, typename SUBNET>
    using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;
    template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
    template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;
    template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
    template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
    template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
    template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
    template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;
    using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
        alevel0<
        alevel1<
        alevel2<
        alevel3<
        alevel4<
        dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2,
        dlib::input_rgb_image_sized<150>
        >>>>>>>>>>>>;


    // ----------------------------------------------------------------------------------------
    /**
     * @brief A face recognition class that provides methods for recognizing faces and managing known faces.
     */
    class FaceRecognizer
    {
    public:

        /**
         * @brief Gets the face feature vector from an image file.
         *
         * @param image_path A string representing the path to the image file.
         * @return The face feature vector as a vector of floats.
         */
        face_features_t GetFaceFeatureFromImg(const std::string& image_path);

        /**
         * @brief Calculates the similarity between a given face feature and an image file.
         *
         * @param face_feature1 A string representing the face feature to compare.
         * @param image_path A string representing the path to the image file to compare against.
         * @return The similarity score between the face feature and the image file, as a float.
         */
        float GetSimilarityFromImg(const std::string& face_feature1, const std::string& image_path);

        /**
         * @brief Recognizes a face from an image file by comparing it against a list of known faces.
         *
         * @param image_path A string representing the path to the image file.
         * @param name The name of the recognized face, passed by reference.
         * @param break_if_first_matched A boolean indicating whether to stop searching for matches after the first match is found.
         * @return An integer indicating the number of matches found.
         */
        int RecognizeImgByKnownFaces(const std::string& image_path, std::string& name, bool break_if_first_matched = true);


        /**
         * @brief Adds a set of known face features to the list of known faces.
         *
         * @param known_face_name The name of the known face.
         * @param face_features The face feature vector as a vector of floats.
         */
        void AddKnownFaceFeatures(std::string known_face_name, face_features_t face_features);

        /**
         * @brief Adds a set of known face features to the list of known faces.
         *
         * @param known_face_name The name of the known face.
         * @param face_features A string representing the face feature vector.
         */
        void AddKnownFaceFeatures(std::string known_face_name, const std::string& face_features);


        /**
         * @brief Gets the face feature vectors from an image file.
         *
         * @param image_path A string representing the path to the image file.
         * @return A vector of face feature vectors, each represented as a vector of floats.
         *
         * This function returns a vector of face feature vectors extracted from the specified image file.
         * This function can be time-consuming, taking 4-500ms to complete.
         */
        // TODO this func spent most time, it takes 4~500ms
        std::vector<face_features_t> GetQueryFaceFeaturesFromImg(const std::string& image_path);

        /**
         * @brief Calculates the similarity between a given face feature and a list of query face features.
         *
         * @param face_feature_str A string representing the face feature to compare.
         * @param query_face_features A vector of query face features, each represented as a vector of floats.
         * @return The similarity score between the face feature and the query face features, as a float.
         */
        float GetSimilarity(const std::string& face_feature_str, std::vector<face_features_t> query_face_features);


        /**
         * @brief Calculates the similarity between a given face feature and a list of query face features.
         *
         * @param face_feature A reference to the face feature to compare.
         * @param query_face_features A vector of query face features, each represented as a vector of floats.
         * @return The similarity score between the face feature and the query face features, as a float.
         */
        float GetSimilarity(face_features_t& face_feature, std::vector<face_features_t> query_face_features);


        /**
         * @brief Checks whether a given face feature matches any of the query face features.
         *
         * @param face_feature_str A string representing the face feature to compare.
         * @param query_face_features A vector of query face features, each represented as a vector of floats.
         * @return A boolean indicating whether a match was found.
         */
        bool Match(const std::string& face_feature_str, std::vector<face_features_t> query_face_features);


        /**
         * @brief Checks whether a given face feature matches any of the query face features.
         *
         * @param face_feature A reference to the face feature to compare.
         * @param query_face_features A vector of query face features, each represented as a vector of floats.
         * @return A boolean indicating whether a match was found.
         */
        bool Match(face_features_t& face_feature, std::vector<face_features_t> query_face_features);

    public:
        /**
         * @brief Encodes a face feature vector as a string.
         *
         * @param face_features The face feature vector as a vector of floats.
         * @return A string representing the encoded face feature vector.
         *
         * This function encodes a face feature vector into a string using a binary format.
         * The resulting string can be stored in a database as a binary blob.
         */
        static std::string EncodeFaceFeature(face_features_t face_features);

        /**
         * @brief Decodes a face feature vector from a string.
         *
         * @param face_features_str A string representing the encoded face feature vector.
         * @return The decoded face feature vector as a vector of floats.
         *
         * This function decodes a face feature vector from a string that was encoded using the EncodeFaceFeature function.
         */
        static face_features_t DecodeFaceFeature(const std::string& face_features_str);


        /**
         * @brief Checks whether a given similarity score indicates a match.
         *
         * @param sim The similarity score to check.
         * @return A boolean indicating whether the similarity score indicates a match.
         * the smaller the better
         */
        static inline bool IsMatched(float sim) { return sim < matched_max_threshold; }

        /**
         * @brief The maximum similarity threshold that indicates a match.
         *
         * This static member variable specifies the maximum similarity threshold that indicates a match.
         * The default value is set to the threshold used by the dlib face recognition model, which achieves
         * an accuracy of 99.38% on the LFW face recognition benchmark.
         */
        static float matched_max_threshold;

    protected:

        dlib::shape_predictor& GetLandmarkDetector() {
            std::call_once(sp_init_flag_, &FaceRecognizer::InitLandmarkDetector, this);
            return sp_;
        }

        anet_type& GetFaceRecognizer() {
            std::call_once(net_init_flag_, &FaceRecognizer::InitFaceRecognizer, this);
            return net_;
        }

    private:
        void InitLandmarkDetector() {
            dlib::deserialize(shape_predictor_dat) >> sp_;
        }

        void InitFaceRecognizer() {
            dlib::deserialize(face_recognition_resnet_dat) >> net_;
        }

        // dat file from  http://dlib.net/files/
        // std::string shape_predictor_dat = "C:/Users/SG220/source/repos/testCmakeProj/shape_predictor_68_face_landmarks.dat";
        // std::string face_recognition_resnet_dat = "C:/Users/SG220/source/repos/testCmakeProj/dlib_face_recognition_resnet_model_v1.dat";
        std::string shape_predictor_dat = "shape_predictor_68_face_landmarks.dat";
        std::string face_recognition_resnet_dat = "dlib_face_recognition_resnet_model_v1.dat";

        dlib::shape_predictor sp_;
        anet_type net_;
        std::once_flag sp_init_flag_;
        std::once_flag net_init_flag_;
        face_features_map_t known_face_features_cache;

        // In principle, there may be multiple faces recognized in the image.
        // The default value of "false" indicates that the recognition will stop after the first face is detected.
        // bool allow_mutil_face_ = false;
    };

}


