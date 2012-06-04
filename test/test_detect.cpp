#include "tpofinder/configure.h"
#include "tpofinder/detect.h"

#include <boost/foreach.hpp>
#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace tpofinder;

class detect : public ::testing::Test {
public:

    virtual void SetUp() {
        models.add(PROJECT_BINARY_DIR + "/data/taco");
        models.add(PROJECT_BINARY_DIR + "/data/blokus");
        detector = Detector(models);
        image = imread(PROJECT_BINARY_DIR + "/data/test/scene-blokus-taco-1.png");
        scene = detector.describe(image);
    }

    Modelbase models;
    Detector detector;
    Mat image;
    Scene scene;

    size_t findIndex(const vector<Detection> detections, const string& modelName) {
        for (size_t i = 0; i < detections.size(); i++) {
            if (detections[i].model.name == modelName) {
                return i;
            }
        }
        return -1;
    }

};

TEST_F(detect, describeSiftCompatible) {
    Feature feature("SIFT", "SIFT", "BruteForce");
    Detector d = Detector(models, feature);
    d.describe(image);
}

TEST_F(detect, describeSurfCompatible) {
    Feature feature("SURF", "SURF", "BruteForce");
    Detector d = Detector(models, feature);
    d.describe(image);
}

TEST_F(detect, sceneHasKeypoints) {
    EXPECT_GE(scene.keypoints.size(), 300);
}

TEST_F(detect, detectTacoInScene) {
    vector<Detection> detections = detector.detect(scene);
    EXPECT_GE(detections.size(), 1);
    EXPECT_GE(findIndex(detections, "taco"), 0);
}

TEST_F(detect, detectTacoInTrainingImage) {
    Scene trainingView = detector.describe(imread(PROJECT_BINARY_DIR + "/data/taco/ref.jpg"));
    vector<Detection> detections = detector.detect(trainingView);
    EXPECT_GE(detections.size(), 1);
    int tacoInd = findIndex(detections, "taco");
    EXPECT_GE(tacoInd, 0);
    EXPECT_NEAR(norm(detections[tacoInd].homography - EYE_HOMOGRAPHY), 0, 0.2);
}

/** For the matching process, each model is regarded as a single image with a
 * collection of keypoints and descriptors. The fact that the keypoints and the
 * descriptors originally came from different images is not transparent to the 
 * matcher. The imgIdx member of DMatch shall be the index of the matched model.
 * If it is required to find the training image of the model where the keypoint
 * was selected in the first place, then accumulating the number of keypoints in
 * the list of training images will allow to recover this piece of information.
 */
TEST_F(detect, modelMatchesImgIdxRefersToModel) {
    vector<Detection> detections = detector.detect(scene);
    for (size_t i = 0; i < detections.size(); i++) {

        BOOST_FOREACH(DMatch& m, detections[i].matches) {
            EXPECT_EQ(m.imgIdx, i);
        }
    }
}

/** Check whether the trainIdx member of a DMatch is within the bounds; minimum
 * is 0 and maximum is allKeypoints.size() of the model it belongs to. */
TEST_F(detect, modelMatchesTrainIdxWithinBounds) {
    vector<Detection> detections = detector.detect(scene);
    for (size_t i = 0; i < detections.size(); i++) {

        BOOST_FOREACH(DMatch& m, detections[i].matches) {
            EXPECT_GE(m.trainIdx, 0);
            EXPECT_LT(m.trainIdx, detections[i].model.allKeypoints.size());
        }
    }
}

TEST_F(detect, eigenvalueFilterIdentity) {
    Detection d;
    d.homography = Mat::eye(3, 3, CV_64FC1);
    EigenvalueFilter f;
    EXPECT_TRUE(f.accept(d));
}

TEST_F(detect, eigenvalueFilterMultipleOfIdentity) {
    EigenvalueFilter f(1. / 3., 3.);

    Detection d1;
    d1.homography = 4 * Mat::eye(3, 3, CV_64FC1);
    EXPECT_FALSE(f.accept(d1));

    Detection d2;
    d2.homography = 2.9 * Mat::eye(3, 3, CV_64FC1);
    EXPECT_TRUE(f.accept(d2));
}

TEST_F(detect, eigenvalueFilterNonSymmetric) {
    EigenvalueFilter f1(1. / 2., 3.0);
    EigenvalueFilter f2(1. / 3., 3.0);
    EigenvalueFilter f3(1. / 3., 1.5);

    Detection d;
    d.homography = (Mat_<double>(3, 3) <<
            2.0, 0.0, 6.0,
            0.0, 0.4, 2.0,
            0.0, 0.0, 1.0);
    EXPECT_FALSE(f1.accept(d));
    EXPECT_TRUE(f2.accept(d));
    EXPECT_FALSE(f3.accept(d));
}

TEST_F(detect, inliersRatioFilterNoMatches) {
    InliersRatioFilter f;
    Detection d;
    EXPECT_FALSE(f.accept(d));
}
