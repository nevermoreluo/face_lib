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

    class FaceRecognizer
    {
    public:

        face_features_t GetFaceFeatureFromImg(const std::string& image_path);


        float GetSimilarityFromImg(const std::string& face_feature1, const std::string& image_path);

        int RecognizeImgByKnownFaces(const std::string& image_path, std::string& name, bool break_if_first_matched = true);

        void AddKnownFaceFeatures(std::string known_face_name, face_features_t face_features);

        void AddKnownFaceFeatures(std::string known_face_name, const std::string& face_features);

        // TODO this func spent most time, it takes 4~500ms
        std::vector<face_features_t> GetQueryFaceFeaturesFromImg(const std::string& image_path);

        float GetSimilarity(const std::string& face_feature_str, std::vector<face_features_t> query_face_features);

        float GetSimilarity(face_features_t& face_feature, std::vector<face_features_t> query_face_features);

        bool Match(const std::string& face_feature_str, std::vector<face_features_t> query_face_features);

        bool Match(face_features_t& face_feature, std::vector<face_features_t> query_face_features);

    public:
        // note buffer type return, be careful with data storage, 
        // Need save the string as a binary blob in the database
        // or use base64 encoding
        // Example pseudocode:
        // db.execute("INSERT INTO faces (name, feature) VALUES (?, ?)", "Alice", face_feature_str.data(), face_feature_str.size());
        static std::string EncodeFaceFeature(face_features_t face_features);

        static face_features_t DecodeFaceFeature(const std::string& face_features_str);


        // the smaller the better
        static inline bool IsMatched(float sim) { return sim < matched_max_threshold; }

        /*
        #   When using a distance threshold of 0.6, the dlib model obtains an accuracy
        #   of 99.38 % on the standard LFW face recognition benchmark, which is
        #   comparable to other state - of - the - art methods for face recognition as of
        #   February 2017. This accuracy means that, when presented with a pair of face
        #   images, the tool will correctly identify if the pair belongs to the same
        #   person or is from different people 99.38 % of the time.
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


