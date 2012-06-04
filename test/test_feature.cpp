#include "tpofinder/configure.h"
#include "tpofinder/feature.h"

#include <gtest/gtest.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace tpofinder;

class feature : public ::testing::Test {
public:

    virtual void SetUp() {
        image = imread(PROJECT_BINARY_DIR + "/data/blokus/ref.jpg", 0);
        ASSERT_FALSE(image.empty());
    }

    void testFeature(const Feature& feature, int minKeypoints) {
        vector<KeyPoint> kpts;
        feature.detector->detect(image, kpts);
        EXPECT_GE(kpts.size(), minKeypoints);
        cv::Mat descs;
        feature.extractor->compute(image, kpts, descs);
    }

    Mat image;

};

/** Check whether the default features work out-of-the box. */
TEST_F(feature, computeDefault) {
    Feature feature;
    testFeature(feature, 300);
}

TEST_F(feature, computeORB) {
    Feature feature("ORB", "ORB", "BruteForce-Hamming");
    testFeature(feature, 300);
}

TEST_F(feature, computeSIFT) {
    Feature feature("SIFT", "SIFT", "BruteForce");
    testFeature(feature, 300);
}

TEST_F(feature, computeSURF) {
    Feature feature("SURF", "SURF", "BruteForce");
    testFeature(feature, 300);
}

/** Check whether we are able to use exotic combinations of feature detectors
 * and descriptor extractors such as ORB feature locations combined with SIFT
 * descriptors. */
TEST_F(feature, computeCustom) {
    Ptr<FeatureDetector> fd = new OrbFeatureDetector(700);
    Ptr<DescriptorExtractor> de = new SiftDescriptorExtractor();
    Ptr<DescriptorMatcher> dm = new BFMatcher(NORM_L2);
    Feature feature(fd, de, dm);
    testFeature(feature, 300);
}

TEST_F(feature, showSIFT) {
    Feature feature("SURF", "SURF", "BruteForce");
    vector<KeyPoint> kpts;
    feature.detector->detect(image, kpts);
    Mat out;
    drawKeypoints(image, kpts, out);
    imshow("feature.showSIFT", out);
}
